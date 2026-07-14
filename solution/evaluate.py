# helper script for the report - compares the task2 and task3 models on
# validation and validation_augmented (FPR and recall). not required for submission.
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from train import (
    CustomCNNDetector,
    ParquetCalibDataset,
    ai_probability,
    evaluation_transform,
)
from train_augmented import StrongerCNNDetector

EVAL_TRANSFORM_224 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def resolve_data_dir(name):
    for c in [os.path.join("data", name),
              os.path.join("..", "data", name),
              os.path.join("..", "AML DATA", name)]:
        if os.path.exists(c):
            return c
    return None

def load_task2_model(ckpt_path):
    model = CustomCNNDetector(channels=32)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model

def load_task3_model(ckpt_path):
    model = StrongerCNNDetector(channels=32)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model

def load_threshold(path, default=0.50):
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    return default

def collect_scores(model, data_dir, transform, tta=False, model_type="task3"):
    ds = ParquetCalibDataset(data_folder=data_dir, img_transformer=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs)
            p = ai_probability(logits)
            if tta:
                flip_logits = model(torch.flip(imgs, dims=[3]))
                p_flip = ai_probability(flip_logits)
                p = (p + p_flip) / 2.0
            scores.extend(p.cpu().numpy().tolist())
            labels.extend(lbls.numpy().tolist())
    return np.array(scores), np.array(labels)

def metrics(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    real = labels == 0
    ai = labels == 1
    fpr = preds[real].mean() if real.sum() > 0 else float('nan')
    recall_ai = preds[ai].mean() if ai.sum() > 0 else float('nan')
    return fpr, recall_ai

def main():
    configs = [
        ("Task 2 model", False, evaluation_transform(), "task2",
         "artifacts/task02/best_model.pt",
         "artifacts/task02/calibrated_threshold.txt"),
        ("Task 3 model", False, EVAL_TRANSFORM_128, "task3",
         "artifacts/task03/best_model_augmented.pt",
         "artifacts/task03/calibrated_threshold.txt"),
        ("Task 3 model + TTA", True, EVAL_TRANSFORM_128, "task3",
         "artifacts/task03/best_model_augmented.pt",
         "artifacts/task03/calibrated_threshold.txt"),
    ]
    splits = [
        ("validation", resolve_data_dir("validation")),
        ("validation_augmented", resolve_data_dir("validation_augmented")),
    ]

    print(f"\n{'Model':<24}{'Split':<24}{'Threshold':<12}{'FPR_real':<12}{'Recall_ai':<12}")
    print("-" * 84)
    for name, tta, transform, model_type, ckpt, th_path in configs:
        if not os.path.exists(ckpt):
            print(f"{name:<24}[checkpoint missing at {ckpt}]")
            continue
            
        if model_type == "task2":
            model = load_task2_model(ckpt)
        else:
            model = load_task3_model(ckpt)
            
        threshold = load_threshold(th_path)
        for split_name, split_dir in splits:
            if split_dir is None:
                print(f"{name:<24}{split_name:<24}[split not found]")
                continue
            scores, labels = collect_scores(
                model, split_dir, transform, tta=tta, model_type=model_type
            )
            fpr, rec = metrics(scores, labels, threshold)
            print(f"{name:<24}{split_name:<24}{threshold:<12.4f}{fpr:<12.4f}{rec:<12.4f}")
    print("-" * 84)
    print("Targets: FPR_real <= 0.20 | Task2 validation recall_ai >= 0.8 | "
          "Task3 augmented recall_ai >= 0.6\n")

if __name__ == "__main__":
    main()
