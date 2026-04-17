from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import os

# Initialize app
app = Flask(__name__)

# ❌ Disable model for deployment
model = None

# Class labels (kept for future use)
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

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Image preprocessing (still kept)
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # 🔥 Handle deployment case
    if model is None:
        return jsonify({
            "disease": "Model not deployed",
            "confidence": 0.0
        })

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

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))