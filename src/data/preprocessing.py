import tensorflow as tf


def preprocess_image(image, target_size):
    """Resize and normalize an image to [0, 1]."""
    image = tf.image.resize(image, [target_size, target_size])
    image = image / 255.0
    return image


def augment_image(image):
    """Apply random augmentations for training."""
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
