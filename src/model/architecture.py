import keras
import tensorflow as tf


# Mapping from config string to (application_module, class).
# Each entry's preprocess_input handles backbone-specific normalization.
_BACKBONES = {
    "ResNet50": (tf.keras.applications.resnet50, tf.keras.applications.ResNet50),
    "ResNet101": (tf.keras.applications.resnet, tf.keras.applications.ResNet101),
    "EfficientNetB0": (tf.keras.applications.efficientnet, tf.keras.applications.EfficientNetB0),
    "EfficientNetB1": (tf.keras.applications.efficientnet, tf.keras.applications.EfficientNetB1),
    "MobileNetV2": (tf.keras.applications.mobilenet_v2, tf.keras.applications.MobileNetV2),
    "VGG16": (tf.keras.applications.vgg16, tf.keras.applications.VGG16),
}


@keras.saving.register_keras_serializable(package="funny_img_decider")
class BackbonePreprocess(tf.keras.layers.Layer):
    """Serializable preprocessing layer: rescales [0,1]→[0,255] then applies
    the backbone's preprocess_input."""

    def __init__(self, backbone_name, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name

    def call(self, inputs):
        x = inputs * 255.0
        app_module = _BACKBONES[self.backbone_name][0]
        return app_module.preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config["backbone_name"] = self.backbone_name
        return config


def _build_custom_cnn(image_size, learning_rate):
    """Build the original custom CNN (unchanged from before)."""
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


def _build_pretrained(image_size, learning_rate, backbone_name,
                      freeze_backbone, classifier_units, classifier_dropout):
    """Build a model with a pretrained backbone + custom classification head."""
    if backbone_name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Supported: {sorted(_BACKBONES.keys())}"
        )

    app_module, backbone_cls = _BACKBONES[backbone_name]
    input_shape = (image_size, image_size, 3)

    # Build the backbone without the classification top.
    backbone = backbone_cls(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = not freeze_backbone

    # Functional API: input → backbone-specific preprocessing → backbone → head
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Rescale [0, 1] → backbone-expected range (baked into model so inference
    # and saved models are self-contained).
    x = BackbonePreprocess(backbone_name, name="backbone_preprocess")(inputs)

    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(classifier_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(classifier_dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(config):
    """Build and return a compiled TensorFlow model for binary humor classification.

    Input:  (batch_size, image_size, image_size, 3) — normalized RGB images [0, 1]
    Output: (batch_size, 1) — sigmoid probability (funny vs not funny)

    If config["model"]["pretrained"] is true, uses a pretrained backbone from
    tf.keras.applications. Otherwise builds the original custom CNN.
    """
    image_size = config["dataset"]["image_size"]
    learning_rate = config["training"]["learning_rate"]
    model_cfg = config.get("model", {})

    if model_cfg.get("pretrained", False):
        return _build_pretrained(
            image_size=image_size,
            learning_rate=learning_rate,
            backbone_name=model_cfg.get("backbone", "ResNet50"),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            classifier_units=model_cfg.get("classifier_units", 128),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.5),
        )
    else:
        return _build_custom_cnn(image_size, learning_rate)
