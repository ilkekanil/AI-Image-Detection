import os
import argparse
import time
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from predict import InferenceImageDataset
from train import ai_probability
from train_augmented import StrongerCNNDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout_seconds', type=int, default=600)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + max(args.timeout_seconds, 1) * 0.95

    out_dir = "artifacts/task03"
    os.makedirs(out_dir, exist_ok=True)

    predict_dir = "data/predict"
    if not os.path.exists(predict_dir):
        predict_dir = os.path.join("..", "data", "predict")
    if not os.path.exists(predict_dir):
        predict_dir = os.path.join("..", "AML DATA", "predict")

    # Re-build architecture and load the best saved checkpoint parameters
    model = StrongerCNNDetector(channels=32)
    ckpt = os.path.join(out_dir, "best_model_augmented.pt")
    if not os.path.exists(ckpt):
        print(f"[ERROR] Task 3 model checkpoint file is missing at: {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    th_path = os.path.join(out_dir, "calibrated_threshold.txt")
    if os.path.exists(th_path):
        with open(th_path, "r") as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.50

    # Ensure inference test files are also processed at 128px
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ids, preds = [], []
    if os.path.exists(predict_dir):
        parquets = sorted([
            os.path.join(predict_dir, f)
            for f in os.listdir(predict_dir)
            if f.endswith('.parquet')
        ])
        for pq in parquets:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "predict_augmented.py reached its time limit; "
                    "no partial predictions were written."
                )
            print(f"-> Predicting labels for file: {os.path.basename(pq)}")
            ds = InferenceImageDataset(parquet_path=pq, target_transform=eval_transform)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            with torch.no_grad():
                for imgs, row_ids in loader:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "predict_augmented.py reached its time limit; "
                            "no partial predictions were written."
                        )
                    logits = model(imgs)
                    ai_scores = ai_probability(logits)
                    decisions = (ai_scores >= threshold).int()
                    ids.extend(row_ids.numpy().tolist())
                    preds.extend(decisions.numpy().tolist())

    # Format output precisely as required by submission guidelines
    df = pd.DataFrame({"row_id": ids, "predicted_label": preds})
    df = df.sort_values(by="row_id")
    out_csv = os.path.join(out_dir, "predictions.csv")
    if time.monotonic() >= deadline:
        raise TimeoutError(
            "predict_augmented.py reached its time limit before output serialization."
        )
    df.to_csv(out_csv, index=False)
    elapsed = time.monotonic() - started
    print(f"-> [SUCCESS] Predictions written to: {out_csv}")
    print(f"-> Total evaluated samples: {len(df)}")
    print(f"-> Prediction runtime: {elapsed:.1f}s / {args.timeout_seconds}s")

if __name__ == "__main__":
    main()
