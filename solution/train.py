import os
import io
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageOps

from prepare import AIImageDataset

class CustomCNNDetector(nn.Module):
    def __init__(self, channels=32):
        super(CustomCNNDetector, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*channels, 2)
        )
    def forward(self, x):
        return self.fc_layer(self.feature_extractor(x))

class ParquetCalibDataset(Dataset):
    def __init__(self, data_folder, img_transformer=None):
        self.img_transformer = img_transformer
        self.pq_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.parquet')])
        
        chunk_list = []
        for file in self.pq_files:
            chunk_df = pd.read_parquet(file, columns=["image", "source_class"])
            chunk_list.append(chunk_df)
        
        if len(chunk_list) > 0:
            self.main_df = pd.concat(chunk_list, ignore_index=True)
        else:
            self.main_df = pd.DataFrame(columns=["image", "source_class"])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, index):
        data_row = self.main_df.iloc[index]
        raw_bytes = data_row['image']
        src_cls = int(data_row['source_class'])
        lbl = 0 if src_cls == 0 else 1
        
        with Image.open(io.BytesIO(raw_bytes)) as source_img:
            source_img = ImageOps.exif_transpose(source_img)
            source_img = source_img.convert("RGB")
            source_img = source_img.resize((224, 224), Image.Resampling.BICUBIC)
            
        if self.img_transformer:
            source_img = self.img_transformer(source_img)
        return source_img, lbl

def run_threshold_calibration(target_model, calib_loader):
    target_model.eval()
    prob_accumulator = []
    
    print("-> Optimizing operational threshold using official calibration data...")
    with torch.no_grad():
        for batch_imgs, batch_lbls in calib_loader:
            net_outputs = target_model(batch_imgs)
            prob_dist = torch.softmax(net_outputs, dim=1)
            ai_scores = prob_dist[:, 1].cpu().numpy()
            
            real_masks = (batch_lbls == 0).numpy()
            prob_accumulator.extend(ai_scores[real_masks])
            
    if len(prob_accumulator) == 0:
        print("[WARN] No real samples located in calibration set. Defaulting to 0.50.")
        return 0.50
        
    computed_th = float(np.percentile(prob_accumulator, 80))
    print(f"-> Calibration successful. Determined decision threshold: {computed_th:.4f}")
    return computed_th

def main():
    tick = time.monotonic()
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--timeout_seconds', type=int, default=1800)
    cmd_args = cli_parser.parse_args()
    
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    
    target_dir = "artifacts/task02"
    os.makedirs(target_dir, exist_ok=True)
    
    labels_csv_path = os.path.join("artifacts", "task01", "cleaned_train", "labels.csv")
    feat_npz_path = os.path.join("artifacts", "classical_train_features.npz")
    
    calib_path = "data/calibration"
    if not os.path.exists(calib_path):
        calib_path = os.path.join("..", "data", "calibration")
    if not os.path.exists(calib_path):
        calib_path = os.path.join("..", "AML DATA", "calibration")

    # 1. Classical Baseline
    if os.path.exists(feat_npz_path):
        binary_data = np.load(feat_npz_path)
        features_x, labels_y = binary_data['X'], binary_data['y']
        baseline_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        baseline_rf.fit(features_x, labels_y)
        print("-> Classical baseline baseline_rf fit sequence completed.")

    # 2. Deep Learning Pipeline
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if os.path.exists(labels_csv_path):
        train_set = AIImageDataset(csv_path=labels_csv_path, transform=img_transforms)
        train_generator = DataLoader(train_set, batch_size=64, shuffle=True)
        
        neural_net = CustomCNNDetector(channels=32)
        loss_fn = nn.CrossEntropyLoss()
        weight_optimizer = optim.AdamW(neural_net.parameters(), lr=1e-3)
        
        max_epochs = 2
        lowest_loss = float('inf')
        
        for ep in range(max_epochs):
            neural_net.train()
            total_loss = 0.0
            for batch_imgs, batch_lbls in train_generator:
                if time.monotonic() - tick > (cmd_args.timeout_seconds * 0.90):
                    break
                weight_optimizer.zero_grad(set_to_none=True)
                predictions = neural_net(batch_imgs)
                batch_loss = loss_fn(predictions, batch_lbls)
                batch_loss.backward()
                weight_optimizer.step()
                total_loss += batch_loss.item()
                
            mean_ep_loss = total_loss / len(train_generator)
            print(f"Epoch [{ep+1}/{max_epochs}] - Mean Loss: {mean_ep_loss:.4f}")
            if mean_ep_loss < lowest_loss:
                lowest_loss = mean_ep_loss
                torch.save(neural_net.state_dict(), os.path.join(target_dir, "best_model.pt"))
    else:
        print("[ERROR] Required dataset labels.csv missing.")
        return

    # 3. Automation of Calibration Point
    if os.path.exists(calib_path) and len([f for f in os.listdir(calib_path) if f.endswith('.parquet')]) > 0:
        try:
            calib_dataset = ParquetCalibDataset(data_folder=calib_path, img_transformer=img_transforms)
            calib_generator = DataLoader(calib_dataset, batch_size=32, shuffle=False)
            final_th = run_threshold_calibration(neural_net, calib_generator)
        except Exception as runtime_err:
            print(f"[WARN] Runtime error encountered: {runtime_err}. Using backup threshold.")
            final_th = 0.50
    else:
        final_th = 0.50
        print("[WARN] Local calibration directory unallocated or empty. Fallback activated.")
        
    with open(os.path.join(target_dir, "calibrated_threshold.txt"), "w") as th_file:
        th_file.write(str(final_th))
    print("=== Pipeline execution sequence finalized successfully ===")

if __name__ == "__main__":
    main()