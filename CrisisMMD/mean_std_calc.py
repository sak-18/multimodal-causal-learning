import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

def calculate_mean_std(image_folder, annotation_file):
    # Load the annotation data
    data = pd.read_csv(annotation_file, sep='\t')

    # Assuming 'image' column contains directory paths relative to the image_folder
    image_paths = data['image'].apply(lambda x: os.path.join(image_folder, x))

    means, stds = [], []
    transform = transforms.ToTensor()  # Convert images to tensors

    for img_path in image_paths:
        if os.path.exists(img_path):
            image = pil_loader(img_path)
            tensor_img = transform(image)
            means.append(tensor_img.mean([1, 2]))  # Mean over H, W for each channel
            stds.append(tensor_img.std([1, 2]))    # Std over H, W for each channel
        else:
            print(f"Warning: {img_path} not found. Skipping.")

    # Calculate global mean and std across all images
    if means:
        mean = torch.stack(means).mean(0)
        std = torch.stack(stds).mean(0)
        
        print(f"Mean: {mean.tolist()}")
        print(f"Std: {std.tolist()}")

        # Save the mean and std to a file
        with open("mean_std_values.txt", "w") as f:
            f.write(f"Mean: {mean.tolist()}\n")
            f.write(f"Std: {std.tolist()}\n")
    else:
        print("No valid images found to calculate mean and std.")

if __name__ == "__main__":
    image_folder = "CrisisMMD_v2.0"  # Update this path to your image folder
    annotation_file = "crisismmd_datasplit_all/task_informative_text_img_train.tsv"  # Update this path to your annotation file

    calculate_mean_std(image_folder, annotation_file)
