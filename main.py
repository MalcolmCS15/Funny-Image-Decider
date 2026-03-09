import argparse
import yaml
from src.data.loader import get_datasets
from src.model.architecture import build_model
from src.training.trainer import train
from src.inference.predict import load_trained_model, predict_image


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Funny Image Decider")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="'train' to train the model, 'predict' to classify an image",
    )
    parser.add_argument(
        "--image", type=str, help="Path to image file (required for predict mode)"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        print("Loading HumorDB dataset...")
        train_ds, val_ds = get_datasets(config)

        print("Building model...")
        model = build_model(config)
        model.summary()
        print("Starting training...")
        train(model, train_ds, val_ds, config)
        print("Training complete. Model saved to:", config["training"]["checkpoint_dir"])

    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        model = load_trained_model(config)
        result = predict_image(model, args.image, config)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']}%")


if __name__ == "__main__":
    main()
