import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image


def find_last_conv_layer(model):
    """Find the name of the last Conv2D layer in a Keras model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")
    return last_conv


def compute_gradcam(model, image_array, conv_layer_name=None):
    """Compute a GradCAM heatmap for a single preprocessed image.

    Args:
        model: A compiled tf.keras.Model.
        image_array: Preprocessed image tensor of shape (H, W, 3) in [0, 1].
        conv_layer_name: Name of conv layer to target. Auto-detects if None.

    Returns:
        heatmap: numpy array of shape (H, W) with values in [0, 1].
    """
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)

    image_batch = tf.expand_dims(tf.cast(image_array, tf.float32), axis=0)

    # Build a functional sub-model for GradCAM that works with Keras 3
    # Sequential models. Create a fresh input and thread it through the layers.
    input_shape = image_array.shape
    inp = tf.keras.Input(shape=input_shape)
    conv_output = None
    x = inp
    for layer in model.layers:
        x = layer(x)
        if layer.name == conv_layer_name:
            conv_output = x
    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        predicted_class = predictions[0, 0]

    grads = tape.gradient(predicted_class, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy()


def overlay_gradcam(pil_image, heatmap, alpha=0.4):
    """Overlay a GradCAM heatmap on a PIL image.

    Args:
        pil_image: Original PIL Image (any size).
        heatmap: 2D numpy array (h, w) with values in [0, 1].
        alpha: Transparency for the heatmap overlay.

    Returns:
        A PIL Image with the heatmap overlaid.
    """
    w, h = pil_image.size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((w, h))
    ) / 255.0

    colormap = cm.jet(heatmap_resized)[:, :, :3]
    colormap = np.uint8(colormap * 255)

    original = np.array(pil_image)
    blended = np.uint8(original * (1 - alpha) + colormap * alpha)
    return Image.fromarray(blended)
