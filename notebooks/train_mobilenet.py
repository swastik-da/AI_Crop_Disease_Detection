import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Fix randomness
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load data
import pickle

with open("model/data.pkl", "rb") as f:
    data, labels = pickle.load(f)

print("Unique labels:", np.unique(labels))
print("Sample labels:", labels[:10])

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Data min/max:", data.min(), data.max())

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Labels fix
labels = labels.astype(int)
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Load MobileNetV2 (WITHOUT top layer)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model/mobilenet_model.keras")

print("MobileNet training complete!")