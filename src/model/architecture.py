import tensorflow as tf


def build_model(config):
    """Build and return a compiled TensorFlow model for binary humor classification.

    Input:  (batch_size, image_size, image_size, 3) — normalized RGB images
    Output: (batch_size, 1) — sigmoid probability (funny vs not funny)
    """
    image_size = config["dataset"]["image_size"]
    learning_rate = config["training"]["learning_rate"]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size, image_size, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, "sigmoid"),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
