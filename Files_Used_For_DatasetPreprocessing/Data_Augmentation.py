import cv2
import albumentations as A
import os
import glob
from pathlib import Path

# Define augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.ColorJitter(p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# Define paths
input_dir = r"C:\Users\Vincent Maes\Documents\PI\Data_3"
output_dir = r"C:\Users\VM\datasets\DetectPot_AUG"
os.makedirs(output_dir, exist_ok=True)

# Function to read and write YOLO label files
def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        bboxes = [list(map(float, line.strip().split()[1:])) for line in file]
    return bboxes

# Remove the distinction between classes and have every class be pothole class. 
def write_yolo_labels(label_path, bboxes):
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            file.write(f"0 {' '.join(map(str, bbox))}\n")

# Loop over images and augment
for img_path in glob.glob(f"{input_dir}/*.jpeg"):
    img = cv2.imread(img_path)
    label_path = img_path.replace('.jpeg', '.txt')
    bboxes = read_yolo_labels(label_path)
    
    # Apply augmentation
    augmented = augment(image=img, bboxes=bboxes, category_ids=[0]*len(bboxes))
    
    # Save augmented image and label
    img_name = Path(img_path).stem + '_aug.jpeg'
    cv2.imwrite(os.path.join(output_dir, img_name), augmented['image'])
    
    label_output_path = os.path.join(output_dir, img_name.replace('.jpeg', '.txt'))
    write_yolo_labels(label_output_path, augmented['bboxes'])
