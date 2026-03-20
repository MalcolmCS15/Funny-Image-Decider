# Funny Image Decider

A binary image classifier that predicts whether an image is **funny or not funny**, trained on the [HumorDB](https://huggingface.co/datasets/kreimanlab/HumorDB) dataset from Kreiman Lab.

Built with TensorFlow/Keras. Supports both a custom CNN trained from scratch and pretrained backbones (transfer learning) for better performance on small datasets.

## Quick Start

```bash
pip install -r requirements.txt
python main.py train
python main.py evaluate
python main.py predict --image path/to/image.jpg
```

## Project Structure

```
├── main.py                        # CLI entry point (train / predict / evaluate)
├── config/
│   └── config.yaml                # All hyperparameters and settings
├── src/
│   ├── model/
│   │   └── architecture.py        # CNN architecture (custom + pretrained)
│   ├── data/
│   │   ├── loader.py              # HuggingFace dataset loading
│   │   └── preprocessing.py       # Image resize, normalize, augment
│   ├── training/
│   │   ├── trainer.py             # Training loop
│   │   └── callbacks.py           # Live plot callback
│   ├── evaluation/
│   │   └── evaluate.py            # Test metrics + GradCAM display
│   ├── inference/
│   │   └── predict.py             # Single-image prediction
│   └── visualization/
│       ├── plots.py               # Loss/accuracy curves
│       └── gradcam.py             # GradCAM heatmap generation
├── checkpoints/                   # Saved model weights
├── outputs/                       # Generated plots and visualizations
└── logs/                          # TensorBoard logs
```

## Configuration

All settings live in `config/config.yaml`.

### Model

```yaml
model:
  pretrained: true           # true = pretrained backbone, false = custom CNN
  backbone: "MobileNetV2"   # ResNet50, EfficientNetB0, MobileNetV2, VGG16
  freeze_backbone: true      # Freeze backbone weights (feature extraction only)
  classifier_units: 128      # Dense layer size in classification head
  classifier_dropout: 0.5    # Dropout rate in classification head
```

**Custom CNN** (`pretrained: false`): A 3-block CNN (32 -> 64 -> 128 filters) with batch normalization, trained entirely from scratch.

**Pretrained backbone** (`pretrained: true`): Loads ImageNet weights from `tf.keras.applications` and adds a classification head on top. Backbone-specific preprocessing is baked into the model so saved models are self-contained.

### Transfer Learning Strategy

For small datasets, start with:
1. `freeze_backbone: true` with learning rate `0.001` — trains only the classification head (~262K params). Fast convergence in 5-10 epochs.
2. If performance plateaus, switch to `freeze_backbone: false` with a lower learning rate (`0.0001`) to fine-tune the entire network.

### Dataset, Training, and Output

```yaml
dataset:
  name: "kreimanlab/HumorDB"
  image_size: 224
  batch_size: 32
  validation_split: 0.2
  seed: 42

training:
  epochs: 10
  learning_rate: 0.001
  early_stopping_patience: 5

output:
  dir: "outputs"
  num_gradcam_samples: 5
  loss_curve: "loss_curve.png"       # Set to "" to skip
  accuracy_curve: "accuracy_curve.png"
  gradcam: "gradcam.png"
```

## Usage

### Train

```bash
python main.py train
```

Trains the model, saves the best checkpoint to `checkpoints/best_model.keras`, and generates live-updating loss/accuracy plots in `outputs/`.

### Evaluate

```bash
python main.py evaluate
```

Runs the saved model on the test split and generates GradCAM visualizations showing what regions the model focuses on.

### Predict

```bash
python main.py predict --image path/to/image.jpg
```

Outputs a label (funny / not funny) and confidence percentage.

### Custom Config

```bash
python main.py --config path/to/custom_config.yaml train
```

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- HuggingFace `datasets`
- A HuggingFace account with access to [kreimanlab/HumorDB](https://huggingface.co/datasets/kreimanlab/HumorDB)

## GradCAM

The evaluate mode generates [GradCAM](https://arxiv.org/abs/1610.02391) heatmaps showing which image regions influenced the model's prediction. Works with both the custom CNN and pretrained backbones.
