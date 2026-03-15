import os

import matplotlib.pyplot as plt


def save_metric_plots(metrics_dict, output_dir="outputs", filenames=None):
    """Save training/validation metric curves from a dict of metric lists.

    Args:
        metrics_dict: Dict mapping metric names to lists of per-epoch values.
        output_dir: Directory to save plot images.
        filenames: Optional dict mapping metric names to output filenames.
                   Keys must match Keras metric names as they appear in
                   model.fit logs (e.g. "loss", "accuracy"). Mismatched keys
                   are silently ignored. If a metric is missing or its value
                   is empty/None, that plot is skipped. If filenames is None,
                   all metrics are saved with default names.

    Returns:
        List of file paths to saved plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    val_keys = {k for k in metrics_dict if k.startswith("val_")}
    train_keys = [k for k in metrics_dict if not k.startswith("val_")]

    saved_paths = []
    for metric in train_keys:
        if filenames is not None:
            filename = filenames.get(metric, "")
            if not filename:
                continue
        else:
            filename = f"{metric}_curve.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(metrics_dict[metric]) + 1)

        ax.plot(epochs, metrics_dict[metric], "o-", label=f"Train {metric}")
        val_metric = f"val_{metric}"
        if val_metric in val_keys:
            ax.plot(epochs, metrics_dict[val_metric], "o-", label=f"Val {metric}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Training & Validation {metric.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
