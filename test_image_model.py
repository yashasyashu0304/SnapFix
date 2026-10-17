import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_output", "image_model_mobilenet.keras")
DATA_DIR = os.path.join(BASE_DIR, "data", "new_images")

IMG_SIZE = 224

CLASS_NAMES = [
    "damaged_road_sign",
    "garbage",
    "graffiti",
    "illegal_parking",
    "pothole_road_crack"
]

CONF_THRESHOLD = 0.60   # minimum confidence
MARGIN_THRESHOLD = 0.15 # top1 - top2 gap
# ----------------------------------------


def load_image(class_name):
    folder = os.path.join(DATA_DIR, class_name)
    fname = sorted(os.listdir(folder))[0]
    path = os.path.join(folder, fname)

    print("Testing image:", path)

    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)

    return arr


def is_prediction_stable(probs, conf_th=CONF_THRESHOLD, margin=MARGIN_THRESHOLD):
    probs = np.array(probs)

    top1_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)[::-1]

    top1 = sorted_probs[0]
    top2 = sorted_probs[1]
    gap = top1 - top2

    if top1 < conf_th:
        return False, "low_confidence"

    if gap < margin:
        return False, "ambiguous"

    return True, "stable"


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
    print("Model output shape:", model.output_shape)

    TEST_CLASS = "pothole_road_crack"  # change to test others
    x = load_image(TEST_CLASS)

    preds = model.predict(x)[0]

    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])

    stable, reason = is_prediction_stable(preds)

    print("\nRaw probabilities:", preds)
    print("Predicted index:", pred_idx)
    print("Predicted class:", pred_class)
    print("Confidence:", round(confidence, 4))

    if stable:
        print("✅ Stable prediction — accept result")
    else:
        print(f"⚠️ Uncertain prediction ({reason}) — require text / human review")
