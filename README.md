# AugmentV1: Recycling Code Detection and Data Augmentation

AugmentV1 is a Python package for detecting recycling codes in images and augmenting image datasets to improve model training for recycling code detection.

## Features

- Load and process image datasets with bounding box annotations
- Extract recycling code regions from images
- Apply various augmentation techniques to images and their annotations:
  - Geometric transformations (flipping, rotation)
  - Color adjustments (brightness, contrast)
  - Noise and blur
  - Cutout/random erasing
  - Packaging label swapping (replace existing labels with new ones)
- Configurable via YAML or JSON configuration files
- Comprehensive logging system
- Command-line interface for common operations

## Installation

### From Source

```bash
git clone https://github.com/AlexanderGrunewald/sherlock_augment.git
cd sherlock_augment
pip install -e .
```

### Using pip

```bash
pip install augmentv1
```

## Quick Start

```python
from augmentv1.data.data_loader import DataLoader
from augmentv1.augmentation.label_augmenter import LabelAugmenter

# Load images and annotations
data_loader = DataLoader(image_dir="data/raw/train")
data_loader.load_images()

# Create an augmenter
augmenter = LabelAugmenter(data_loader)

# Augment a single image
for fname, img, bboxes in data_loader.image_dataset:
    augmented_img, augmented_bboxes = augmenter.augment_labels(img, bboxes)
    # Use augmented image and bounding boxes...
    break

# Augment the entire dataset
augmented_dataset = augmenter.augment_dataset(augmentations_per_image=3)
```

## Command-line Interface

AugmentV1 provides a command-line interface for common operations:

```bash
# Create a default configuration file
augmentv1 config create --output config.yaml

# Load and display dataset summary
augmentv1 data summary --image-dir data/raw/train

# Augment a dataset
augmentv1 augment --image-dir data/raw/train --output-dir data/augmented
```

## Configuration

AugmentV1 can be configured using YAML or JSON files. Create a default configuration file:

```bash
augmentv1 config create --output config.yaml
```

Example configuration:

```yaml
data:
  image_dir: data/raw/train
  label_dir: data/raw/train
  image_ext: png
  label_ext: txt

augmentation:
  enabled: true
  techniques:
    - horizontal_flip
    - vertical_flip
    - rotation
    - brightness
    - contrast
    - swap_labels
  augmentations_per_image: 3
# To be implemented
model:
  architecture: yolo
  input_size: [416, 416]
  batch_size: 16
  epochs: 100

logging:
  level: INFO
  file: logs/augmentv1.log
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
