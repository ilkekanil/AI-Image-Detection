import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AIImageDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_column="binary_label"):

        self.metadata_df = pd.read_csv(csv_path)
        self.transform = transform
        if label_column not in self.metadata_df.columns:
            raise ValueError(f"Unknown label column: {label_column}")
        self.label_column = label_column

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, index):
        data_row = self.metadata_df.iloc[index]
        target_path = data_row['image_path']
        
        img_instance = Image.open(target_path).convert('RGB')
        target_label = int(data_row[self.label_column])

        if self.transform:
            img_instance = self.transform(img_instance)
            
        return img_instance, target_label



def compute_color_statistics(csv_source, npz_destination):
    print(f"-> Extracting classical engineered features from: {csv_source}...")
    if not os.path.exists(csv_source):
        print(f"[ERROR] Source registry {csv_source} could not be located.")
        return
        
    source_df = pd.read_csv(csv_source)
    extracted_features = []
    mapped_labels = []
    
    for _, current_row in source_df.iterrows():
        img_path = current_row['image_path']
        if not os.path.exists(img_path):
            continue
            
        with Image.open(img_path).convert('RGB') as visual_file:
            matrix_representation = np.array(visual_file)
        
      
        avg_r, avg_g, avg_b = matrix_representation[:,:,0].mean(), matrix_representation[:,:,1].mean(), matrix_representation[:,:,2].mean()
        dev_r, dev_g, dev_b = matrix_representation[:,:,0].std(), matrix_representation[:,:,1].std(), matrix_representation[:,:,2].std()
        
        
        distribution_r, _ = np.histogram(matrix_representation[:,:,0], bins=8, range=(0, 255))
        distribution_g, _ = np.histogram(matrix_representation[:,:,1], bins=8, range=(0, 255))
        distribution_b, _ = np.histogram(matrix_representation[:,:,2], bins=8, range=(0, 255))
        
        combined_vector = np.hstack([avg_r, avg_g, avg_b, dev_r, dev_g, dev_b, distribution_r, distribution_g, distribution_b])
        extracted_features.append(combined_vector)
        mapped_labels.append(int(current_row['binary_label'])) 
            
    feature_matrix = np.array(extracted_features)
    label_vector = np.array(mapped_labels)
    
    os.makedirs(os.path.dirname(npz_destination), exist_ok=True)
    np.savez(npz_destination, X=feature_matrix, y=label_vector)
    print(f"-> Feature engineering matrix saved successfully at: {npz_destination}")



# Pipeline entry point

if __name__ == "__main__":
    
    source_csv = os.path.join("artifacts", "task01", "cleaned_train", "labels.csv")
    target_npz = os.path.join("artifacts", "classical_train_features.npz") 
    
    compute_color_statistics(source_csv, target_npz)
    print("=== Data Feature Preparation Sequence Terminated ===")
