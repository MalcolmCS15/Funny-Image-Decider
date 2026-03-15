import os

import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import get_test_samples
from src.inference.predict import predict_pil
from src.visualization.gradcam import compute_gradcam, overlay_gradcam


def evaluate_model(model, test_ds):
    """Evaluate a model on a tf.data.Dataset and return metrics.

    Args:
        model: A compiled tf.keras.Model.
        test_ds: A batched tf.data.Dataset of (image, label) pairs.

    Returns:
        dict with keys matching model.metrics_names
        (e.g. {"loss": 0.71, "accuracy": 0.52}).
    """
    results = model.evaluate(test_ds, verbose=1)
    return dict(zip(model.metrics_names, results))


def display_sample_predictions(model, samples, image_size, threshold=0.5,
                               output_dir="outputs", output_filename="gradcam.png"):
    """Generate a figure with original images, GradCAM overlays, and predictions.

    Args:
        model: A compiled, built tf.keras.Model with Conv2D layers.
        samples: List of (PIL.Image, int_label) tuples.
        image_size: Target size for preprocessing (int).
        threshold: Confidence threshold for "funny" classification.
        output_dir: Directory to save the composite figure.
        output_filename: Filename for the saved figure. If empty, no-op.

    Returns:
        Path to the saved figure, or None if output_filename is empty.
    """
    if not output_filename:
        return None

    os.makedirs(output_dir, exist_ok=True)

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (pil_image, true_label) in enumerate(samples):
        prob, pred_label, preprocessed = predict_pil(
            model, pil_image, image_size, threshold
        )

        heatmap = compute_gradcam(model, preprocessed)
        overlay = overlay_gradcam(pil_image, heatmap)

        axes[i, 0].imshow(pil_image)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("GradCAM")
        axes[i, 1].axis("off")

        true_str = "Funny" if true_label == 1 else "Not Funny"
        pred_str = "Funny" if pred_label == 1 else "Not Funny"
        correct = "CORRECT" if true_label == pred_label else "WRONG"
        color = "green" if true_label == pred_label else "red"

        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.5, 0.5,
            f"True: {true_str}\nPred: {pred_str}\nConf: {prob:.2%}\n{correct}",
            ha="center", va="center", fontsize=14,
            color=color, fontweight="bold",
            transform=axes[i, 2].transAxes,
        )

    fig.suptitle("Sample Predictions with GradCAM", fontsize=16)
    fig.tight_layout()
    path = os.path.join(output_dir, output_filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def run_gradcam_outputs(model, config, hf_ds=None):
    """Generate GradCAM visualization if configured. No-op if gradcam filename is empty.

    Args:
        model: A compiled tf.keras.Model.
        config: Project config dict.
        hf_ds: Optional pre-loaded HuggingFace DatasetDict to avoid re-downloading.

    Returns:
        Path to saved figure, or None if skipped.
    """
    output_cfg = config.get("output", {})
    gradcam_filename = output_cfg.get("gradcam", "")
    if not gradcam_filename:
        return None

    output_dir = output_cfg.get("dir", "outputs")
    n_samples = output_cfg.get("num_gradcam_samples", 5)

    print(f"\nGenerating GradCAM for {n_samples} random test samples...")
    samples = get_test_samples(config, n=n_samples, hf_ds=hf_ds)
    fig_path = display_sample_predictions(
        model, samples,
        image_size=config["dataset"]["image_size"],
        threshold=config["inference"]["confidence_threshold"],
        output_dir=output_dir,
        output_filename=gradcam_filename,
    )
    print(f"  Saved: {fig_path}")
    return fig_path
