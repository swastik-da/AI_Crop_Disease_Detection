import os
import cv2
import numpy as np

# Path to dataset
dataset_path = "dataset"

# Image size
IMG_SIZE = 224

data = []
labels = []

# Get all disease folders
classes = os.listdir(dataset_path)

print("Total Classes:", len(classes))

for label, disease in enumerate(classes):
    class_path = os.path.join(dataset_path, disease)
    
    # Skip if not a folder
    if not os.path.isdir(class_path):
        continue
    
    print(f"Processing: {disease}")
    
    images = os.listdir(class_path)
    
    for img_name in images[:100]:  # limit for speed (IMPORTANT)
        img_path = os.path.join(class_path, img_name)
        
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize (0–255 → 0–1)
        img = img / 255.0
        
        data.append(img)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("\nData shape:", data.shape)
print("Labels shape:", labels.shape)
import pickle
import os

os.makedirs("model", exist_ok=True)

with open("model/data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print("Data saved successfully!")