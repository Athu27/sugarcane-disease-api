from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "model/model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1j3jFUsfpXx_pICMHdP1eW_HO3u171EtL"  # replace with actual file ID

# Download model if not exists
os.makedirs("model", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
LABELS = ["BrownRust", "Dry", "Healthy", "Mawa", "Mites", "RedSpot", "YellowLeaf"]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    image_bytes = file.read()
    image_tensor = preprocess_image(image_bytes)
    predictions = model.predict(image_tensor)
    predicted_index = np.argmax(predictions)
    predicted_label = LABELS[predicted_index]
    return jsonify({"prediction": predicted_label})
