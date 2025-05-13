import os
import humanize
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Literal
from dataclasses import dataclass
import numpy as np

@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    cls: int

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h
        yield self.cls

@dataclass
class PixelBoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2
        yield self.cls


class LabelDataset:
    """Handles loading and storing label data."""

    def __init__(self, label_dir: Union[str, Path]):
        self.label_dir = Path(label_dir)
        self.labels: List[str] = []

    def load_labels(self) -> List[str]:
        """Loads labels from the label directory. To be implemented."""
        raise NotImplementedError("Label loading not implemented.")

    def get_random_label(self) -> str:
        if not self.labels:
            raise ValueError("Labels have not been loaded yet.")
        return random.choice(self.labels)


class ImageDataset:
    """Handles loading images and corresponding bounding box annotations."""

    def __init__(self, image_dir: Union[str, Path], ext_img="png", ext_bbx="txt"):
        self.image_dir: Path = Path(image_dir)
        self.ext_img = ext_img
        self.ext_bbx = ext_bbx
        self.images: List[Tuple[str, any]] = []
        self.bboxes: Dict[str, List[BoundingBox|PixelBoundingBox]] = {}
        self.image_shapes: Dict[str, Tuple[int, int]] = {}
        self.bbox_format: Literal["yolo", "pixel"] = "yolo"

    def load_images(self) -> List[Tuple[str, any]]:
        """Loads all images with the specified extension."""
        self.images = []
        for file in self.image_dir.glob(f"*.{self.ext_img}"):
            img = cv2.imread(str(file))
            if img is not None:
                h, w = img.shape[:2]
                self.image_shapes[file.name] = (h, w)
                self.images.append((file.name, img))
        return self.images

    def load_bounding_boxes(self) ->  Dict[str, List[BoundingBox]]:
        """Loads bounding boxes from associated text files."""
        self.bboxes = {}
        for file in self.image_dir.glob(f"*.{self.ext_bbx}"):
            img_name = file.stem + f".{self.ext_img}"
            with open(file, "r") as f:
                boxes = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append(BoundingBox(x, y, w, h, int(cls)))
                self.bboxes[img_name] = boxes
        return self.bboxes

    def get_image(self, name: str) -> Optional[any]:
        """Retrieve image by filename."""
        return next((img for fname, img in self.images if fname == name), None)

    def get_bounding_boxes(self, name: str) -> List[BoundingBox]:
        """Retrieve YOLO-format BoundingBox objects for a given image filename."""
        return self.bboxes.get(name, [])

    def denormalize_bounding_boxes_(self) -> None:
        """Convert YOLO-normalized BoundingBox instances to absolute PixelBoundingBox and update self.bboxes."""
        if not self.images:
            raise ValueError("Images must be loaded before denormalizing bounding boxes.")
        if self.is_pixel_format():
            raise Warning("Image already in pixel format.")

        for img_name, img in self.images:
            h, w = self.image_shapes[img_name]
            boxes = self.bboxes.get(img_name, [])
            abs_boxes = []
            for box in boxes:
                x1 = int((box.x - box.w / 2) * w)
                y1 = int((box.y - box.h / 2) * h)
                x2 = int((box.x + box.w / 2) * w)
                y2 = int((box.y + box.h / 2) * h)
                abs_boxes.append(PixelBoundingBox(x1, y1, x2, y2, box.cls))
            self.bboxes[img_name] = abs_boxes
        self.bbox_format = "pixel"

    def normalize_bounding_boxes(self) -> None:
        """Convert PixelBoundingBox instances to YOLO-normalized BoundingBox format and update self.bboxes."""
        if not self.images:
            raise ValueError("Images must be loaded before normalizing bounding boxes.")
        if self.is_yolo_format():
            raise Warning("Image already in YOLO format.")

        for img_name, img in self.images:
            h, w= self.image_shapes[img_name]
            boxes = self.bboxes.get(img_name, [])
            norm_boxes = []
            for box in boxes:
                x_center = ((box.x1 + box.x2) / 2) / w
                y_center = ((box.y1 + box.y2) / 2) / h
                box_w = (box.x2 - box.x1) / w
                box_h = (box.y2 - box.y1) / h
                norm_boxes.append(BoundingBox(x_center, y_center, box_w, box_h, box.cls))
            self.bboxes[img_name] = norm_boxes
        self.bbox_format = "yolo"

    def is_yolo_format(self) -> bool:
        return self.bbox_format == "yolo"

    def is_pixel_format(self) -> bool:
        return self.bbox_format == "pixel"

    def __getitem__(self, idx: int) -> Tuple[str, any, list]:
        fname, img = self.images[idx]
        bboxes = self.get_bounding_boxes(fname)

        if self.is_yolo_format():
            assert all(isinstance(b, BoundingBox) for b in bboxes)
        elif self.is_pixel_format():
            assert all(isinstance(b, PixelBoundingBox) for b in bboxes)

        return fname, img, bboxes

    def __len__(self) -> int:
        return len(self.images)


class DataLoader:
    """Coordinates loading of image and label datasets."""

    def __init__(self, image_dir: Optional[str] = None, label_dir: Optional[str] = None):
        self.image_dataset: Optional[ImageDataset] = ImageDataset(image_dir) if image_dir else None
        self.label_dataset: Optional[LabelDataset] = LabelDataset(label_dir) if label_dir else None

        self.images: Optional[List[Tuple[str, any]]] = None
        self.bboxes: Optional[Dict[str, List[Tuple[float, float, float, float]]]] = None
        self.labels: Optional[List[str]] = None

    def load_data(self) -> None:
        """Load both images and labels."""
        if not self.image_dataset or not self.label_dataset:
            raise ValueError("Both image and label datasets must be initialized.")
        self.images = self.image_dataset.load_images()
        self.bboxes = self.image_dataset.load_bounding_boxes()
        self.labels = self.label_dataset.load_labels()

    def load_images(self) -> None:
        """Load only images and bounding boxes."""
        if self.image_dataset is None:
            raise ValueError("ImageDataset not initialized.")
        self.images = self.image_dataset.load_images()
        self.bboxes = self.image_dataset.load_bounding_boxes()

    def load_labels(self) -> None:
        """Load only labels."""
        if not self.label_dataset:
            raise ValueError("LabelDataset not initialized.")
        self.labels = self.label_dataset.load_labels()

    def get_images(self) -> Optional[List[Tuple[str, any]]]:
        return self.images

    def get_labels(self) -> Optional[List[str]]:
        return self.labels

    def query(self, by: str, value) -> list:
        """
        Query image or bounding box data by class or filename.

        Args:
            by (str): Either "class" or "filename".
            value: Class ID (int) or filename (str) depending on the query type.

        Returns:
            List of matching bounding boxes, image crops, or metadata.
        """
        if not self.image_dataset:
            raise ValueError("Image dataset not loaded.")

        if by == "class":
            results = []
            for fname, img, boxes in self.image_dataset:
                for box in boxes:
                    if box.cls == value:
                        results.append((fname, box))
            return results

        elif by == "filename":
            for fname, img, boxes in self.image_dataset:
                if fname == value:
                    return [(box.cls, box) for box in boxes], img
            return []

        else:
            raise ValueError("Invalid query type. Use 'class' or 'filename'.")

    def summary(self, show_class_names: bool = False) -> None:
        """
        Prints a summary of the dataset including class counts, number of files, total size, etc.

        Args:
            show_class_names (bool): Whether to show class names from the LabelDataset.
        """
        if not self.image_dataset:
            raise ValueError("Image dataset not loaded.")

        if not self.image_dataset.images:
            print("No images loaded.")
            return

        print("📊 Dataset Summary")
        print("-" * 40)

        # Number of files
        num_images = len(self.image_dataset.images)
        print(f"🖼️  Number of images: {num_images}")

        # Total file size
        total_bytes = sum(
            os.path.getsize(self.image_dataset.image_dir / fname)
            for fname, _ in self.image_dataset.images
        )
        print(f"💾 Total size: {humanize.naturalsize(total_bytes)}")

        # Class distribution
        class_counts = defaultdict(int)
        for boxes in self.image_dataset.bboxes.values():
            for box in boxes:
                class_counts[box.cls] += 1

        print(f"🧾 Unique classes: {len(class_counts)}")
        print("📈 Bounding boxes per class:")
        for cls, count in sorted(class_counts.items()):
            line = f"    Class {cls}: {count} boxes"
            if show_class_names and self.label_dataset and self.label_dataset.labels:
                class_name = self.label_dataset.labels[int(cls)] if int(cls) < len(self.label_dataset.labels) else "Unknown"
                line += f" ({class_name})"
            print(line)

        # Bounding box totals
        total_boxes = sum(class_counts.values())
        print(f"📦 Total bounding boxes: {total_boxes}")

        print("-" * 40)


class RecyclingCodeExtractor:
    def __init__(self, image_dataset: ImageDataset):
        self.image_dataset = image_dataset
        self.class_to_crops: dict[str, list[any]] = {}
        if image_dataset.is_yolo_format():
            image_dataset.denormalize_bounding_boxes_()

    def extract_code(self):
        for fname, img, boxes in self.image_dataset:
            for box in boxes:
                x1, y1, x2, y2, cls = box
                cropped = img[y1:y2, x1:x2]

                if cls not in self.class_to_crops:
                    self.class_to_crops[cls] = []

                self.class_to_crops[cls].append((fname, cropped))

    def display_class(self, cls: int, n: int = None, cols: int = 5, figsize: tuple = (15, 8)) -> None:
        """
        Display up to `n` cropped recycling code images belonging to a specific class in a grid.
        """
        items = self.class_to_crops.get(cls, [])

        if not items:
            print(f"No crops found for class {cls}.")
            return

        if n is not None:
            items = random.sample(items, min(n, len(items)))  # safely limit sample size

        rows = math.ceil(len(items) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, (np.ndarray, list)) else [axes]

        for i, ax in enumerate(axes):
            if i < len(items):
                fname, crop = items[i]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
                ax.set_title(fname, fontsize=9)
            ax.axis("off")

        plt.suptitle(f"Class {cls} – showing {len(items)} crop(s)", fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    img_dir = r"C:\Users\alexg\Documents\Work\bosch\augmentv1\data\raw\train"
    data_loader = DataLoader(image_dir=img_dir)
    data_loader.load_images()
    code_extractor = RecyclingCodeExtractor(data_loader.image_dataset)
    code_extractor.extract_code()
    for i in range(12):
        code_extractor.display_class(i, n=10)
