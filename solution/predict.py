import os
import io
import argparse
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from train import CustomCNNDetector, ai_probability, evaluation_transform

class InferenceImageDataset(Dataset):
    def __init__(self, parquet_path, target_transform=None):
        raw_dataframe = pd.read_parquet(parquet_path)
        
        raw_dataframe.columns = [col.strip().replace(" ", "_") for col in raw_dataframe.columns]
        
        self.dataset_matrix = raw_dataframe[["row_id", "image"]].copy()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_matrix)

    def __getitem__(self, index):
        data_item = self.dataset_matrix.iloc[index]
        row_identifier = int(data_item["row_id"])
        compressed_bytes = data_item["image"]
        
        with Image.open(io.BytesIO(compressed_bytes)) as active_image:
            active_image = ImageOps.exif_transpose(active_image)
            active_image = active_image.convert("RGB")
            active_image = active_image.resize((224, 224), Image.Resampling.BICUBIC)
            
        if self.target_transform:
            active_image = self.target_transform(active_image)
        return active_image, row_identifier

def main():
    cli_args_parser = argparse.ArgumentParser()
    cli_args_parser.add_argument('--timeout_seconds', type=int, default=600)
    parsed_runtime_args = cli_args_parser.parse_args()
    started = time.monotonic()
    deadline = started + max(parsed_runtime_args.timeout_seconds, 1) * 0.95

    destination_artifacts_dir = "artifacts/task02"
    os.makedirs(destination_artifacts_dir, exist_ok=True)
    
    inference_source_dir = "data/predict" 
    if not os.path.exists(inference_source_dir):
        inference_source_dir = os.path.join("..", "data", "predict")
    if not os.path.exists(inference_source_dir):
        inference_source_dir = os.path.join("..", "AML DATA", "predict")

    evaluation_engine = CustomCNNDetector(channels=32)
    checkpoint_binary_path = os.path.join(destination_artifacts_dir, "best_model.pt")
    
    if os.path.exists(checkpoint_binary_path):
        evaluation_engine.load_state_dict(torch.load(checkpoint_binary_path, map_location='cpu'))
    else:
        print(f"[ERROR] Trained target weights missing at {checkpoint_binary_path}.")
        return

    evaluation_engine.eval()

    th_registry_path = os.path.join(destination_artifacts_dir, "calibrated_threshold.txt")
    if os.path.exists(th_registry_path):
        with open(th_registry_path, "r") as th_stream:
            calibrated_operating_point = float(th_stream.read().strip())
    else:
        calibrated_operating_point = 0.50

    # This is the exact deterministic preprocessing used for calibration and
    # held-out validation in train.py.
    evaluation_tensor_transforms = evaluation_transform()

    collected_identifiers = []
    generated_predictions = []

    if os.path.exists(inference_source_dir):
        discovered_parquets = sorted([
            os.path.join(inference_source_dir, current_file) 
            for current_file in os.listdir(inference_source_dir) 
            if current_file.endswith('.parquet')
        ])
        
        for targeted_parquet in discovered_parquets:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "predict.py reached its time limit; no partial predictions were written."
                )
            print(f"-> Processing current evaluation target: {os.path.basename(targeted_parquet)}")
            runtime_dataset = InferenceImageDataset(parquet_path=targeted_parquet, target_transform=evaluation_tensor_transforms) 
            runtime_loader = DataLoader(runtime_dataset, batch_size=32, shuffle=False)

            with torch.no_grad():
                for process_images, process_ids in runtime_loader:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "predict.py reached its time limit; no partial predictions were written."
                        )
                    raw_logits = evaluation_engine(process_images)
                    synthetic_class_probabilities = ai_probability(raw_logits)
                    
                    binary_decisions = (synthetic_class_probabilities >= calibrated_operating_point).int()
                    
                    collected_identifiers.extend(process_ids.numpy().tolist())
                    generated_predictions.extend(binary_decisions.numpy().tolist())

    submission_dataframe = pd.DataFrame({
        "row_id": collected_identifiers,
        "predicted_label": generated_predictions
    })
    
    submission_dataframe = submission_dataframe.sort_values(by="row_id")
    
    final_output_csv = os.path.join(destination_artifacts_dir, "predictions.csv")
    if time.monotonic() >= deadline:
        raise TimeoutError(
            "predict.py reached its time limit before output serialization."
        )
    submission_dataframe.to_csv(final_output_csv, index=False)
    elapsed = time.monotonic() - started
    print(f"-> [SUCCESS] Verification data compiled cleanly at: {final_output_csv}")
    print(f"-> Total evaluated instances committed: {len(submission_dataframe)}")
    print(f"-> Prediction runtime: {elapsed:.1f}s / {parsed_runtime_args.timeout_seconds}s")

if __name__ == "__main__":
    main()
