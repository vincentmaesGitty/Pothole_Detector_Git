# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:11:02 2024

@author: Vincent Maes
"""

import os
import random
import shutil
from pathlib import Path

# Set paths
source_folder = r"C:\Users\VM\datasets\DetectPot_AUG"  # Folder with all images and labels
output_folder = r"C:\Users\VM\datasets\DetectPot_AUG_Split"  # Output folder for YOLOv5 format
train_pct = 0.85  # Percentage of images for training
val_pct = 0.07    # Percentage of images for validation
test_pct = 0.08   # Percentage of images for testing

# Set seed for reproducibility
random.seed(42)

# Make sure percentages add up to 1
assert round(train_pct + val_pct + test_pct, 2) == 1, "Percentages must add up to 1!"

# Create the output folder structure
for split in ["train", "valid", "test"]:
    os.makedirs(Path(output_folder) / split / "images", exist_ok=True)
    os.makedirs(Path(output_folder) / split / "labels", exist_ok=True)

# List all images and labels
image_files = list(Path(source_folder).glob("*.jpeg"))
label_files = list(Path(source_folder).glob("*.txt"))

# Sort to maintain pairing
image_files.sort()
label_files.sort()

# Shuffle and split data
data_size = len(image_files)
indices = list(range(data_size))
random.shuffle(indices)

train_end = int(train_pct * data_size)
val_end = train_end + int(val_pct * data_size)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Helper function to move files
def move_files(indices, split):
    for i in indices:
        img_file = image_files[i]
        label_file = label_files[i]
        shutil.copy(img_file, Path(output_folder) / split / "images" / img_file.name)
        shutil.copy(label_file, Path(output_folder) / split / "labels" / label_file.name)

# Move files into the split folders
move_files(train_indices, "train")
move_files(val_indices, "valid")
move_files(test_indices, "test")

print("Dataset successfully split and organized for YOLOv5!")
