from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("model/mobilenet_model.keras")

# Class labels (IMPORTANT — match your dataset order)
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_YellowLeaf_Curl_Virus"
]

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# API route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file.stream)

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "disease": class_names[predicted_class].replace("_", " "),
        "confidence": confidence
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True)