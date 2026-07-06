import os
import io
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from prepare import AIImageDataset
from train import ParquetCalibDataset

# Task 2 baseline was failing on distortions, so adding more capacity here.
# Added BatchNorm and some dropout to stop it from overfitting immediately.
class StrongerCNNDetector(nn.Module):
    def __init__(self, channels=32, dropout=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, 2*channels, 3, padding=1), nn.BatchNorm2d(2*channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2*channels, 4*channels, 3, padding=1), nn.BatchNorm2d(4*channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(4*channels, 8*channels, 3, padding=1), nn.BatchNorm2d(8*channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(8*channels, 2*channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2*channels, 2),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

class RandomJPEGCompression:
    # Need real JPEG artifacts to break the frequency signatures of AI images
    def __init__(self, quality_range=(60, 90), p=0.15):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            return reopened.convert("RGB")

def calibrate_threshold_conservative(model, calib_loader, target_fpr=0.18):
    # Keep it at 18% target to be safe against the strict 20% limit in validation
    model.eval()
    real_scores = []
    with torch.no_grad():
        for imgs, lbls in calib_loader:
            probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            real_scores.extend(probs[(lbls == 0).numpy()])
    if len(real_scores) == 0:
        return 0.50
    return float(np.percentile(real_scores, 100 * (1 - target_fpr)))

def resolve_data_dir(name):
    # Just checking different path variants for local vs cluster runs
    candidates = [
        os.path.join("data", name),
        os.path.join("..", "data", name),
        os.path.join("..", "AML DATA", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def main():
    tick = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout_seconds', type=int, default=1800)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    # Fixing seeds for reproducibility requirement
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    out_dir = "artifacts/task03"
    os.makedirs(out_dir, exist_ok=True)

    labels_csv = os.path.join("artifacts", "task01", "cleaned_train", "labels.csv")
    task03_ckpt = os.path.join(out_dir, "best_model_augmented.pt")

    if not os.path.exists(labels_csv):
        print(f"[ERROR] Cleaned training labels missing at {labels_csv}.")
        return

    model = StrongerCNNDetector(channels=32, dropout=0.3)
    torch.save(model.state_dict(), task03_ckpt)

    labels_df = pd.read_csv(labels_csv)
    counts = labels_df['binary_label'].value_counts()
    print(f"-> Class balance: real={int(counts.get(0, 1))}, ai={int(counts.get(1, 1))}")

    # Dropping resolution to 128x128 for efficient retraining within the 1800s CPU budget
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGCompression(quality_range=(60, 90), p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = AIImageDataset(csv_path=labels_csv, transform=aug_transform)

    if args.quick:
        subset_n = max(1, int(len(train_set) * 0.2))
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(train_set), generator=gen)[:subset_n].tolist()
        train_set = torch.utils.data.Subset(train_set, idx)
        print(f"-> QUICK MODE: running training loop on a {subset_n} sample subset.")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    max_epochs = 8
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    stop = False

    for ep in range(max_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for imgs, lbls in train_loader:
            # Shifted to 0.97 to utilize full MacBook clock time without cutting too early
            if time.monotonic() - tick > args.timeout_seconds * 0.97:
                print("-> Approaching timeout limit, saving and stopping.")
                stop = True
                break
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        if n_batches > 0:
            mean_loss = running / n_batches
            print(f"Epoch [{ep+1}/{max_epochs}] - Mean Loss: {mean_loss:.4f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), task03_ckpt)
                print(f"   Saved improved checkpoint (loss={mean_loss:.4f})")
        if stop:
            break

    # Evaluation transforms must match the 128px training shape
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.load_state_dict(torch.load(task03_ckpt, map_location='cpu'))
    calib_aug_dir = resolve_data_dir("calibration_augmented")
    threshold = 0.50
    
    if os.path.exists(calib_aug_dir) and any(f.endswith('.parquet') for f in os.listdir(calib_aug_dir)):
        try:
            calib_ds = ParquetCalibDataset(data_folder=calib_aug_dir, img_transformer=eval_transform)
            calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
            threshold = calibrate_threshold_conservative(model, calib_loader, target_fpr=0.19)
            print(f"-> Calculated operating point (Target FPR=19%): {threshold:.4f}")
        except Exception as err:
            print(f"Automatic calibration failed ({err}). Using baseline threshold 0.50.")
    else:
        print("calibration_augmented folder missing or empty. Using baseline threshold 0.50.")

    with open(os.path.join(out_dir, "calibrated_threshold.txt"), "w") as f:
        f.write(str(threshold))

    print("=== Task 3 training pipeline finished ===")

if __name__ == "__main__":
    main()