import os

import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
from src.data.preprocessing import preprocess_image, augment_image


def _authenticate_hf():
    """Authenticate with Hugging Face using token from env or stored login."""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    # If no env token, huggingface_hub falls back to ~/.huggingface/token
    # (set via `huggingface-cli login`)


def load_humordb(config):
    """Load the full HumorDB DatasetDict from Hugging Face.

    Returns:
        A HuggingFace DatasetDict with 'train', 'validation', and 'test' splits.
    """
    _authenticate_hf()
    return load_dataset(config["dataset"]["name"])


def _make_tf_dataset(hf_dataset, config, training=False):
    """Convert a Hugging Face dataset split into a tf.data.Dataset."""
    image_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]
    shuffle_buffer = config.get("training", {}).get("shuffle_buffer_size", 1000)

    def generator():
        for example in hf_dataset:
            image = example["image"]
            label = example["binary_rating"]
            image = np.array(image.convert("RGB"), dtype=np.float32)
            yield image, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    ds = ds.map(
        lambda img, lbl: (preprocess_image(img, image_size), lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.map(
            lambda img, lbl: (augment_image(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(config):
    """Main entry point: returns (train_ds, val_ds) as tf.data.Datasets."""
    hf_ds = load_humordb(config)
    train_data = hf_ds["train"]
    val_data = hf_ds["validation"]
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")
    train_ds = _make_tf_dataset(train_data, config, training=True)
    val_ds = _make_tf_dataset(val_data, config, training=False)
    return train_ds, val_ds


def get_test_dataset(config, hf_ds=None):
    """Load the HumorDB test split as a tf.data.Dataset.

    Args:
        config: Project config dict.
        hf_ds: Optional pre-loaded HuggingFace DatasetDict to avoid re-downloading.
    """
    if hf_ds is None:
        hf_ds = load_humordb(config)
    test_data = hf_ds["test"]
    print(f"Test: {len(test_data)} examples")
    return _make_tf_dataset(test_data, config, training=False)


def get_test_samples(config, n=5, seed=42, hf_ds=None):
    """Return n random (PIL.Image, label) pairs from the test split.

    Args:
        config: Project config dict.
        n: Number of samples to return.
        seed: Random seed for reproducibility.
        hf_ds: Optional pre-loaded HuggingFace DatasetDict to avoid re-downloading.
    """
    if hf_ds is None:
        hf_ds = load_humordb(config)
    test_data = hf_ds["test"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_data), size=n, replace=False)
    samples = []
    for idx in indices:
        example = test_data[int(idx)]
        pil_image = example["image"].convert("RGB")
        label = example["binary_rating"]
        samples.append((pil_image, label))
    return samples
