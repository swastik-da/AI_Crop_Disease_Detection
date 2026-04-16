import os
import cv2
import matplotlib.pyplot as plt

# CHANGE THIS if needed
dataset_path = "dataset"

# If your dataset has 'train' folder, use this:
if "train" in os.listdir(dataset_path):
    dataset_path = os.path.join(dataset_path, "train")

classes = os.listdir(dataset_path)

print("Total Classes:", len(classes))
print("Sample Classes:", classes[:5])

sample_class = classes[0]
class_path = os.path.join(dataset_path, sample_class)

images = os.listdir(class_path)

plt.figure(figsize=(10,5))

for i in range(5):
    img_path = os.path.join(class_path, images[i])
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis('off')

plt.show()