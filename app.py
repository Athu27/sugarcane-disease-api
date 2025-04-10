from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = tf.keras.models.load_model("model/model.h5")

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
