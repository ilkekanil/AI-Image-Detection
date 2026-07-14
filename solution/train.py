"""
Trains and calibrates the Task 2 AI-image detector on CPU.

We use the calibration split to pick the best model checkpoint and operating threshold. 
The validation split is strictly held out until the very end, serving only to verify 
if we met our target operating point (AI recall >= 0.8 and real-image FPR <= 0.2).
"""

import argparse
import io
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from prepare import AIImageDataset


TASK2_IMAGE_SIZE = 128
TARGET_CALIBRATION_FPR = 0.15  # Buffer margin to stay safely below the hard 20% limit


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class CustomCNNDetector(nn.Module):
    """Compact source-family CNN with an explicit residual input branch."""

    def __init__(self, channels=32, dropout=0.25, num_classes=6):
        super().__init__()
        # Keeping a local residual helps catch high-frequency AI artifacts 
        # without losing the broader RGB content.
        self.feature_extractor = nn.Sequential(
            ConvBlock(6, channels),
            ConvBlock(channels, 2 * channels),
            ConvBlock(2 * channels, 4 * channels),
            ConvBlock(4 * channels, 8 * channels),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(8 * channels, 2 * channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, num_classes),
        )

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        residual = x - local_mean
        return self.fc_layer(self.feature_extractor(torch.cat((x, residual), dim=1)))


class ParquetCalibDataset(Dataset):
    def __init__(self, data_folder, img_transformer=None):
        self.img_transformer = img_transformer
        self.pq_files = sorted(
            os.path.join(data_folder, name)
            for name in os.listdir(data_folder)
            if name.endswith(".parquet")
        )
        chunks = [
            pd.read_parquet(path, columns=["image", "source_class"])
            for path in self.pq_files
        ]
        self.main_df = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=["image", "source_class"])
        )

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, index):
        row = self.main_df.iloc[index]
        label = 0 if int(row["source_class"]) == 0 else 1
        with Image.open(io.BytesIO(row["image"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BICUBIC)
        if self.img_transformer:
            image = self.img_transformer(image)
        return image, label


def resolve_data_dir(name):
    candidates = [
        os.path.join("data", name),
        os.path.join("..", "data", name),
        os.path.join("..", "AML DATA", name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize((TASK2_IMAGE_SIZE, TASK2_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def collect_scores(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            batch_scores = ai_probability(model(images))
            scores.extend(batch_scores.cpu().numpy().tolist())
            labels.extend(batch_labels.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def ai_probability(logits):
    """Convert six source-family logits to the required binary AI score."""
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[:, 1:].sum(dim=1)


def threshold_for_max_fpr(scores, labels, target_fpr=TARGET_CALIBRATION_FPR):
    """Find the threshold that keeps the real-image FPR strictly under our target."""
    real_scores = np.sort(scores[labels == 0])
    if len(real_scores) == 0:
        raise ValueError("Calibration data contains no real images.")

    allowed_false_positives = int(math.floor(target_fpr * len(real_scores)))
    if allowed_false_positives == 0:
        return float(np.nextafter(real_scores[-1], np.inf))

    # Use the raw boundary if it gives us the exact count we want. 
    # If tie scores push us over the FPR cap, bump the threshold up slightly.
    boundary = real_scores[-allowed_false_positives]
    if int(np.sum(real_scores >= boundary)) <= allowed_false_positives:
        return float(boundary)
    return float(np.nextafter(boundary, np.inf))


def classification_metrics(scores, labels, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    real = labels == 0
    ai = labels == 1
    false_positives = int(np.sum(predictions[real] == 1))
    true_positives = int(np.sum(predictions[ai] == 1))
    return {
        "samples": int(len(labels)),
        "real_samples": int(np.sum(real)),
        "ai_samples": int(np.sum(ai)),
        "threshold": float(threshold),
        "false_positives": false_positives,
        "false_positive_rate": float(false_positives / np.sum(real)),
        "true_positives": true_positives,
        "recall_ai": float(true_positives / np.sum(ai)),
        "accuracy": float(np.mean(predictions == labels)),
    }


def run_threshold_calibration(target_model, calib_loader, target_fpr=TARGET_CALIBRATION_FPR):
    print(f"-> Calibrating threshold at empirical FPR <= {target_fpr:.0%}...")
    scores, labels = collect_scores(target_model, calib_loader)
    threshold = threshold_for_max_fpr(scores, labels, target_fpr)
    metrics = classification_metrics(scores, labels, threshold)
    print(
        f"-> Calibration: threshold={threshold:.4f}, "
        f"FPR={metrics['false_positive_rate']:.4f}, "
        f"AI recall={metrics['recall_ai']:.4f}"
    )
    return threshold, metrics


def make_parquet_loader(split_name, transform, batch_size=64):
    split_dir = resolve_data_dir(split_name)
    if split_dir is None:
        return None
    dataset = ParquetCalibDataset(split_dir, img_transformer=transform)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def fit_classical_baseline(feature_path):
    """Retain the required second model family for the report comparison."""
    if not os.path.exists(feature_path):
        print("[WARN] Classical feature file missing; run prepare.py first.")
        return
    data = np.load(feature_path)
    baseline = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    baseline.fit(data["X"], data["y"])
    print("-> Classical Random Forest comparison model fitted.")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    output_dir = os.path.join("artifacts", "task02")
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join("artifacts", "task01", "cleaned_train", "labels.csv")
    feature_path = os.path.join("artifacts", "classical_train_features.npz")
    checkpoint_path = os.path.join(output_dir, "best_model.pt")

    fit_classical_baseline(feature_path)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Required cleaned labels are missing: {labels_path}")

    calibration_loader = make_parquet_loader("calibration", evaluation_transform())
    if calibration_loader is None:
        raise FileNotFoundError("The required data/calibration split is missing or empty.")

    labels_frame = pd.read_csv(labels_path)
    counts = labels_frame["binary_label"].value_counts()
    real_count = int(counts.get(0, 0))
    ai_count = int(counts.get(1, 0))
    if real_count == 0 or ai_count == 0:
        raise ValueError("Training data must contain both real and AI images.")
    print(f"-> Training samples: real={real_count}, AI={ai_count}")

    training_transform = transforms.Compose([
        transforms.Resize((TASK2_IMAGE_SIZE, TASK2_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    train_dataset = AIImageDataset(
        labels_path,
        transform=training_transform,
        label_column="source_class",
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = CustomCNNDetector(channels=32)
    # Since the 6 source classes are balanced, using an auxiliary binary loss 
    # keeps the model focused on the main goal: separating real vs. AI.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    binary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([math.sqrt(ai_count / real_count), 1.0])
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_recall = -1.0
    best_calibration = None
    best_epoch = 0
    torch.save(model.state_dict(), checkpoint_path)

    # Cut off training early to leave enough time for calibration, validation, and saving files.
    training_deadline = started + args.timeout_seconds * 0.85
    stop_training = False
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, batch_labels in train_loader:
            if time.monotonic() >= training_deadline:
                print("-> Training time reserve reached; finalizing best checkpoint.")
                stop_training = True
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            classification_loss = criterion(logits, batch_labels)
            binary_labels = (batch_labels != 0).long()
            binary_logits = torch.stack(
                (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)), dim=1
            )
            loss = classification_loss + 0.25 * binary_criterion(
                binary_logits, binary_labels
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1

        if batches == 0:
            break
        scheduler.step()
        mean_loss = running_loss / batches
        threshold, calibration_metrics = run_threshold_calibration(
            model, calibration_loader, TARGET_CALIBRATION_FPR
        )
        current_recall = calibration_metrics["recall_ai"]
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] loss={mean_loss:.4f}, "
            f"calibration recall={current_recall:.4f}"
        )
        if current_recall > best_recall:
            best_recall = current_recall
            best_calibration = calibration_metrics
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print("   Saved improved FPR-constrained checkpoint.")
        if stop_training:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    final_threshold, final_calibration = run_threshold_calibration(
        model, calibration_loader, TARGET_CALIBRATION_FPR
    )
    with open(os.path.join(output_dir, "calibrated_threshold.txt"), "w", encoding="utf-8") as stream:
        stream.write(f"{final_threshold:.17g}\n")

    summary = {
        "protocol": {
            "checkpoint_selection_split": "calibration",
            "threshold_selection_split": "calibration",
            "validation_used_for_tuning": False,
            "target_calibration_fpr": TARGET_CALIBRATION_FPR,
            "required_validation_fpr_max": 0.20,
            "target_validation_recall_ai": 0.80,
        },
        "best_epoch": best_epoch,
        "calibration": final_calibration,
    }

    validation_loader = make_parquet_loader("validation", evaluation_transform())
    if validation_loader is not None:
        validation_scores, validation_labels = collect_scores(model, validation_loader)
        validation_metrics = classification_metrics(
            validation_scores, validation_labels, final_threshold
        )
        validation_metrics["fpr_constraint_satisfied"] = (
            validation_metrics["false_positive_rate"] <= 0.20
        )
        validation_metrics["recall_target_satisfied"] = (
            validation_metrics["recall_ai"] >= 0.80
        )
        summary["validation"] = validation_metrics
        print(
            "-> HELD-OUT VALIDATION: "
            f"FPR={validation_metrics['false_positive_rate']:.4f}, "
            f"AI recall={validation_metrics['recall_ai']:.4f}"
        )
        if not validation_metrics["fpr_constraint_satisfied"]:
            print("[WARN] The strict validation FPR <= 0.20 requirement was not met.")
        if not validation_metrics["recall_target_satisfied"]:
            print("[WARN] The target validation AI recall >= 0.80 was not met.")
    else:
        print("[WARN] Validation split unavailable; independent verification was skipped.")

    summary["runtime_seconds"] = time.monotonic() - started
    with open(os.path.join(output_dir, "validation_metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print("=== Task 2 training and constrained calibration completed ===")


if __name__ == "__main__":
    main()
