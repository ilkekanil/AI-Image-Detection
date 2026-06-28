import os
import io
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from train import CustomCNNDetector

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

    evaluation_tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    collected_identifiers = []
    generated_predictions = []

    if os.path.exists(inference_source_dir):
        discovered_parquets = sorted([
            os.path.join(inference_source_dir, current_file) 
            for current_file in os.listdir(inference_source_dir) 
            if current_file.endswith('.parquet')
        ])
        
        for targeted_parquet in discovered_parquets:
            print(f"-> Processing current evaluation target: {os.path.basename(targeted_parquet)}")
            runtime_dataset = InferenceImageDataset(parquet_path=targeted_parquet, target_transform=evaluation_tensor_transforms) 
            runtime_loader = DataLoader(runtime_dataset, batch_size=32, shuffle=False)

            with torch.no_grad():
                for process_images, process_ids in runtime_loader:
                    raw_logits = evaluation_engine(process_images)
                    softmax_prob_spread = torch.softmax(raw_logits, dim=1)
                    synthetic_class_probabilities = softmax_prob_spread[:, 1]
                    
                    binary_decisions = (synthetic_class_probabilities >= calibrated_operating_point).int()
                    
                    collected_identifiers.extend(process_ids.numpy().tolist())
                    generated_predictions.extend(binary_decisions.numpy().tolist())

    submission_dataframe = pd.DataFrame({
        "row_id": collected_identifiers,
        "predicted_label": generated_predictions
    })
    
    submission_dataframe = submission_dataframe.sort_values(by="row_id")
    
    final_output_csv = os.path.join(destination_artifacts_dir, "predictions.csv")
    submission_dataframe.to_csv(final_output_csv, index=False)
    print(f"-> [SUCCESS] Verification data compiled cleanly at: {final_output_csv}")
    print(f"-> Total evaluated instances committed: {len(submission_dataframe)}")

if __name__ == "__main__":
    main()