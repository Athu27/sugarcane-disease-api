from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("sugarcane_disease_model_with_attention.h5")

class_names = ['Red Spot', 'Brown Rust', 'Yellow Leaf', 'Myts', 'Eriosoma Lanigerum', 'Dry Leaf', 'Healthy Leaf']

@app.route("/")
def home():
    return "Sugarcane Disease Detection API"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))  # Update size if needed
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    return jsonify({
        "prediction": predicted_class,
        "confidence": float(np.max(predictions[0]))
    })
