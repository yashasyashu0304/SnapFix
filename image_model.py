import os
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_output", "image_model_mobilenet.keras")

# Order must match train_image_model.py
CLASS_NAMES = [
    "damaged_concrete_structures",
    "damaged_electric_poles",
    "damaged_road_sign",
    "fallen_trees",
    "garbage",
    "graffiti",
    "illegal_parking",
    "no_electricity",
    "pothole_road_crack",
    "water_logging"
]

IMG_SIZE = 224

def load_image_model(model_path: str = MODEL_PATH):
    """Load the trained 2‑class image model."""
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path: str):
    """Load an image file and prepare it for the model."""
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image_class(model, image_path: str):
    """Return (predicted_label, confidence_float) for a single image."""
    x = preprocess_image(image_path)
    probs = model.predict(x)[0]
    print("Raw probs:", probs)
    print("Argmax index:", np.argmax(probs))
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    return label, confidence

if __name__ == "__main__":
    model = load_image_model()
    model.summary()
