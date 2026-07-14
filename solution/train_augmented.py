"""Task 3 augmentation fine-tuning with constrained checkpoint selection."""

import argparse
import io
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from prepare import AIImageDataset
from train import (
    CustomCNNDetector,
    ParquetCalibDataset,
    classification_metrics,
    collect_scores,
    evaluation_transform,
    threshold_for_max_fpr,
)


TARGET_AUGMENTED_FPR = 0.18


class StrongerCNNDetector(CustomCNNDetector):
    """Task 2 source-family detector continued with robustness augmentation."""

    def __init__(self, channels=32, dropout=0.25):
        super().__init__(channels=channels, dropout=dropout, num_classes=6)


class RandomJPEGCompression:
    def __init__(self, quality_range=(55, 92), probability=0.25):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(*self.quality_range),
        )
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            return reopened.convert("RGB")


def resolve_data_dir(name):
    candidates = [
        os.path.join("data", name),
        os.path.join("..", "data", name),
        os.path.join("..", "AML DATA", name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def make_split_loader(name, transform, batch_size=64):
    directory = resolve_data_dir(name)
    if not os.path.isdir(directory):
        return None
    dataset = ParquetCalibDataset(directory, img_transformer=transform)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def calibrate_model(model, loader, target_fpr=TARGET_AUGMENTED_FPR):
    scores, labels = collect_scores(model, loader)
    threshold = threshold_for_max_fpr(scores, labels, target_fpr)
    metrics = classification_metrics(scores, labels, threshold)
    print(
        f"-> Augmented calibration: threshold={threshold:.4f}, "
        f"FPR={metrics['false_positive_rate']:.4f}, "
        f"AI recall={metrics['recall_ai']:.4f}"
    )
    return threshold, metrics


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--calibrate_only",
        action="store_true",
        help="Reuse the saved Task 3 checkpoint and regenerate evaluation artifacts.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    output_dir = os.path.join("artifacts", "task03")
    os.makedirs(output_dir, exist_ok=True)
    task2_checkpoint = os.path.join("artifacts", "task02", "best_model.pt")
    task3_checkpoint = os.path.join(output_dir, "best_model_augmented.pt")
    labels_path = os.path.join("artifacts", "task01", "cleaned_train", "labels.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Cleaned training labels are missing: {labels_path}")

    model = StrongerCNNDetector(channels=32, dropout=0.25)
    if args.calibrate_only:
        if not os.path.exists(task3_checkpoint):
            raise FileNotFoundError(f"Task 3 checkpoint is missing: {task3_checkpoint}")
        model.load_state_dict(
            torch.load(task3_checkpoint, map_location="cpu", weights_only=True)
        )
        print("-> CALIBRATE-ONLY MODE: reusing the saved Task 3 checkpoint.")
    else:
        if not os.path.exists(task2_checkpoint):
            raise FileNotFoundError(
                f"Task 2 checkpoint is required for robust fine-tuning: {task2_checkpoint}"
            )
        model.load_state_dict(
            torch.load(task2_checkpoint, map_location="cpu", weights_only=True)
        )
        # The Task 2 checkpoint is a passing fallback even if Docker cannot
        # complete a fine-tuning epoch.
        torch.save(model.state_dict(), task3_checkpoint)
        print("-> Initialized Task 3 from the verified Task 2 checkpoint.")

    eval_transform = evaluation_transform()
    calibration_loader = make_split_loader(
        "calibration_augmented", eval_transform, batch_size=64
    )
    if calibration_loader is None:
        raise FileNotFoundError("data/calibration_augmented is missing or empty.")

    best_threshold, best_calibration = calibrate_model(
        model, calibration_loader, TARGET_AUGMENTED_FPR
    )
    best_recall = best_calibration["recall_ai"]
    best_epoch = 0

    labels_frame = pd.read_csv(labels_path)
    counts = labels_frame["binary_label"].value_counts()
    real_count = int(counts.get(0, 0))
    ai_count = int(counts.get(1, 0))
    print(f"-> Training samples: real={real_count}, AI={ai_count}")

    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.82, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGCompression(quality_range=(55, 92), probability=0.25),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
            p=0.20,
        ),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if not args.calibrate_only:
        training_dataset = AIImageDataset(
            labels_path,
            transform=augmentation,
            label_column="source_class",
        )
        if args.quick:
            subset_size = max(1, int(0.20 * len(training_dataset)))
            subset_generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(
                len(training_dataset), generator=subset_generator
            )[:subset_size].tolist()
            training_dataset = torch.utils.data.Subset(training_dataset, indices)
            print(f"-> QUICK MODE: {subset_size} training samples.")

        loader_generator = torch.Generator().manual_seed(42)
        training_loader = DataLoader(
            training_dataset,
            batch_size=128,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
        )
        source_criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
        binary_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([math.sqrt(ai_count / real_count), 1.0])
        )
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
        training_deadline = started + max(args.timeout_seconds, 1) * 0.80

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            batches = 0
            complete_epoch = True
            for images, source_labels in training_loader:
                if time.monotonic() >= training_deadline:
                    complete_epoch = False
                    print("-> Time reserve reached; discarding the partial epoch.")
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                binary_labels = (source_labels != 0).long()
                binary_logits = torch.stack(
                    (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)),
                    dim=1,
                )
                loss = source_criterion(logits, source_labels)
                loss = loss + 0.25 * binary_criterion(binary_logits, binary_labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batches += 1

            if not complete_epoch:
                break
            scheduler.step()
            threshold, calibration = calibrate_model(
                model, calibration_loader, TARGET_AUGMENTED_FPR
            )
            mean_loss = running_loss / max(1, batches)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] loss={mean_loss:.4f}, "
                f"constrained recall={calibration['recall_ai']:.4f}"
            )
            if calibration["recall_ai"] > best_recall:
                best_recall = calibration["recall_ai"]
                best_threshold = threshold
                best_calibration = calibration
                best_epoch = epoch + 1
                torch.save(model.state_dict(), task3_checkpoint)
                print("   Saved improved augmented checkpoint.")

    model.load_state_dict(
        torch.load(task3_checkpoint, map_location="cpu", weights_only=True)
    )
    best_threshold, best_calibration = calibrate_model(
        model, calibration_loader, TARGET_AUGMENTED_FPR
    )
    with open(
        os.path.join(output_dir, "calibrated_threshold.txt"),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(f"{best_threshold:.17g}\n")

    summary = {
        "protocol": {
            "initial_checkpoint": "artifacts/task02/best_model.pt",
            "checkpoint_selection_split": "calibration_augmented",
            "threshold_selection_split": "calibration_augmented",
            "validation_used_for_tuning": False,
            "target_calibration_fpr": TARGET_AUGMENTED_FPR,
            "required_validation_fpr_max": 0.20,
            "target_validation_augmented_recall_ai": 0.60,
            "maximum_fine_tuning_epochs": args.epochs,
        },
        "selected_fine_tuning_epoch": best_epoch,
        "calibration_augmented": best_calibration,
    }

    for split_name in ("validation", "validation_augmented"):
        split_loader = make_split_loader(split_name, eval_transform, batch_size=64)
        if split_loader is None:
            continue
        scores, labels = collect_scores(model, split_loader)
        metrics = classification_metrics(scores, labels, best_threshold)
        metrics["fpr_constraint_satisfied"] = metrics["false_positive_rate"] <= 0.20
        if split_name == "validation_augmented":
            metrics["recall_target_satisfied"] = metrics["recall_ai"] >= 0.60
        summary[split_name] = metrics
        print(
            f"-> HELD-OUT {split_name}: "
            f"FPR={metrics['false_positive_rate']:.4f}, "
            f"AI recall={metrics['recall_ai']:.4f}"
        )

    summary["runtime_seconds"] = time.monotonic() - started
    with open(
        os.path.join(output_dir, "validation_metrics.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print("=== Task 3 augmentation pipeline completed ===")


if __name__ == "__main__":
    main()
