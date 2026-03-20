import numpy as np
import tensorflow as tf
from PIL import Image
from src.data.preprocessing import preprocess_image
import src.model.architecture  # noqa: F401 — registers custom layers for model loading


def load_trained_model(config):
    """Load a trained model from the checkpoint path."""
    model_path = config["inference"]["model_path"]
    return tf.keras.models.load_model(model_path)


def predict_pil(model, pil_image, image_size, threshold=0.5):
    """Run inference on a PIL Image.

    Args:
        model: A compiled tf.keras.Model.
        pil_image: A PIL Image (RGB).
        image_size: Target size for preprocessing (int).
        threshold: Confidence threshold for "funny" classification.

    Returns:
        Tuple of (probability, predicted_label) where probability is a float
        in [0, 1] and predicted_label is 0 or 1.
    """
    img_array = np.array(pil_image, dtype=np.float32)
    preprocessed = preprocess_image(img_array, image_size)
    prob = float(model.predict(tf.expand_dims(preprocessed, 0), verbose=0)[0][0])
    pred_label = 1 if prob >= threshold else 0
    return prob, pred_label, preprocessed


def predict_image(model, image_path, config):
    """Predict whether an image is funny or not.

    Returns:
        dict with keys:
            - "label": "funny" or "not funny"
            - "confidence": float between 0 and 100 (percentage)
    """
    image_size = config["dataset"]["image_size"]
    threshold = config["inference"]["confidence_threshold"]

    pil_image = Image.open(image_path).convert("RGB")
    prob, pred_label, _ = predict_pil(model, pil_image, image_size, threshold)

    label = "funny" if pred_label == 1 else "not funny"
    confidence = prob if pred_label == 1 else 1.0 - prob

    return {
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
    }
