import os
import tensorflow as tf


def train(model, train_ds, val_ds, config):
    """Train the model with early stopping and checkpointing."""
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    log_dir = config["training"]["log_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history
