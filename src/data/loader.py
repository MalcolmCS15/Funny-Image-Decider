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
    """Load HumorDB dataset from Hugging Face and return tf.data.Datasets."""
    _authenticate_hf()

    dataset_name = config["dataset"]["name"]
    seed = config["dataset"]["seed"]
    val_split = config["dataset"]["validation_split"]

    ds = load_dataset(dataset_name)

    train_data = ds["train"]
    val_data = ds["validation"]

    return train_data, val_data


def _make_tf_dataset(hf_dataset, config, training=False):
    """Convert a Hugging Face dataset split into a tf.data.Dataset."""
    image_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]

    def generator():
        for example in hf_dataset:
            image = example["image"]
            label = example["binary_rating"]
            # Convert PIL image to numpy array
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

    # if training:
    #     ds = ds.map(
    #         lambda img, lbl: (augment_image(img), lbl),
    #         num_parallel_calls=tf.data.AUTOTUNE,
    #     )
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(config):
    """Main entry point: returns (train_ds, val_ds) as tf.data.Datasets."""
    train_data, val_data = load_humordb(config)
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")
    train_ds = _make_tf_dataset(train_data, config, training=True)
    val_ds = _make_tf_dataset(val_data, config, training=False)
    return train_ds, val_ds
