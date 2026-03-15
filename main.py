import argparse
import matplotlib
matplotlib.use("Agg")
import yaml
from src.data.loader import get_datasets, get_test_dataset, load_humordb
from src.model.architecture import build_model
from src.training.trainer import train
from src.inference.predict import load_trained_model, predict_image
from src.evaluation.evaluate import evaluate_model, run_gradcam_outputs


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Funny Image Decider")
    parser.add_argument(
        "mode",
        choices=["train", "predict", "evaluate"],
        help="'train', 'predict', or 'evaluate'",
    )
    parser.add_argument(
        "--image", type=str, help="Path to image file (required for predict mode)"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output", {}).get("dir", "outputs")

    if args.mode == "train":
        print("Loading HumorDB dataset...")
        train_ds, val_ds = get_datasets(config)

        print("Building model...")
        model = build_model(config)
        model.summary()
        print("Starting training...")
        history = train(model, train_ds, val_ds, config)
        print("Training complete. Model saved to:", config["training"]["checkpoint_dir"])
        print("Training plots saved live to:", output_dir)

    elif args.mode == "evaluate":
        print("Loading trained model...")
        model = load_trained_model(config)

        print("Loading HumorDB dataset...")
        hf_ds = load_humordb(config)

        print("Loading test dataset...")
        test_ds = get_test_dataset(config, hf_ds=hf_ds)

        print("Evaluating on test set...")
        metrics = evaluate_model(model, test_ds)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        run_gradcam_outputs(model, config, hf_ds=hf_ds)

    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        model = load_trained_model(config)
        result = predict_image(model, args.image, config)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']}%")


if __name__ == "__main__":
    main()
