import numpy as np
import tensorflow as tf
from PIL import Image
from src.data.preprocessing import preprocess_image


def load_trained_model(config):
    """Load a trained model from the checkpoint path."""
    model_path = config["inference"]["model_path"]
    return tf.keras.models.load_model(model_path)


def predict_image(model, image_path, config):
    """Predict whether an image is funny or not.

    Returns:
        dict with keys:
            - "label": "funny" or "not funny"
            - "confidence": float between 0 and 100 (percentage)
    """
    image_size = config["dataset"]["image_size"]

    image = Image.open(image_path).convert("RGB")
    image = np.array(image, dtype=np.float32)
    image = preprocess_image(image, image_size)
    image = tf.expand_dims(image, axis=0)

    probability = model.predict(image, verbose=0)[0][0]

    threshold = config["inference"]["confidence_threshold"]
    is_funny = probability >= threshold

    label = "funny" if is_funny else "not funny"
    confidence = probability if is_funny else 1.0 - probability

    return {
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
    }
