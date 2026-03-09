import tensorflow as tf


def build_model(config):
    """Build and return a compiled TensorFlow model for binary humor classification.

    YOUR TASK: Define the model architecture below.

    Input:  (batch_size, image_size, image_size, 3) — normalized RGB images
    Output: (batch_size, 1) — sigmoid probability (funny vs not funny)

    The training pipeline expects:
      - A compiled tf.keras.Model
      - Binary crossentropy loss (handled by trainer if not compiled here)
      - Output in [0, 1] range (use sigmoid activation on final layer)
    """
    image_size = config["dataset"]["image_size"]
    learning_rate = config["training"]["learning_rate"]

    # -------------------------------------------------------------------------
    # TODO: Define your model architecture here.
    #
    # Example skeleton (replace with your own design):
    #
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(image_size, image_size, 3)),
    #     # ... your layers here ...
    #     tf.keras.layers.Dense(1, activation="sigmoid"),
    # ])
    # -------------------------------------------------------------------------

    raise NotImplementedError(
        "Define your model architecture in src/model/architecture.py"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
