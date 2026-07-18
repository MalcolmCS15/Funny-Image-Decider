import os
import tensorflow as tf
from src.training.callbacks import LivePlotCallback


def train(model, train_ds, val_ds, config):
    """Train the model with early stopping and checkpointing."""
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    log_dir = config["training"]["log_dir"]

    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("dir", "outputs")
    plot_filenames = {
        "loss": output_cfg.get("loss_curve", ""),
        "accuracy": output_cfg.get("accuracy_curve", ""),
    }

    model_cfg = config.get("model", {})
    if model_cfg.get("pretrained", False):
        run_name = model_cfg.get("backbone", "pretrained")
    else:
        run_name = "custom_cnn"
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_best.keras")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        LivePlotCallback(output_dir=output_dir, filenames=plot_filenames),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history, checkpoint_path
