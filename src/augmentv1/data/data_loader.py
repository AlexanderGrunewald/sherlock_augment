"""
Data loading classes for image datasets and annotations.

This module provides classes for loading and managing image datasets
and their annotations.
"""

import os
import humanize
from collections import defaultdict
import random
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Literal, Any

import cv2
import numpy as np

from augmentv1.data.bounding_box import BoundingBox, PixelBoundingBox


class ImageDataset:
    """
    Handles loading images and corresponding bounding box annotations.
    
    This class provides functionality to load images and their corresponding
    bounding box annotations, and to convert between different bounding box formats.
    
    Attributes:
        image_dir (Path): Directory containing images and annotations
        ext_img (str): File extension for images
        ext_bbx (str): File extension for bounding box annotations
        images (List[Tuple[str, Any]]): List of loaded images with their filenames
        bboxes (Dict[str, List[BoundingBox|PixelBoundingBox]]): Dictionary mapping filenames to bounding boxes
        image_shapes (Dict[str, Tuple[int, int]]): Dictionary mapping filenames to image shapes (height, width)
        bbox_format (Literal["yolo", "pixel"]): Format of the bounding boxes
    """

    def __init__(self, image_dir: Union[str, Path], ext_img: str = "png", ext_bbx: str = "txt"):
        """
        Initialize the ImageDataset.
        
        Args:
            image_dir (Union[str, Path]): Directory containing images and annotations
            ext_img (str, optional): File extension for images. Defaults to "png".
            ext_bbx (str, optional): File extension for bounding box annotations. Defaults to "txt".
        """
        self.image_dir: Path = Path(image_dir)
        self.ext_img = ext_img
        self.ext_bbx = ext_bbx
        self.images: List[Tuple[str, Any]] = []
        self.bboxes: Dict[str, List[Union[BoundingBox, PixelBoundingBox]]] = {}
        self.image_shapes: Dict[str, Tuple[int, int]] = {}
        self.bbox_format: Literal["yolo", "pixel"] = "yolo"

    def load_images(self) -> List[Tuple[str, Any]]:
        """
        Load all images with the specified extension.
        
        Returns:
            List[Tuple[str, Any]]: List of tuples containing (filename, image)
        """
        self.images = []
        for file in self.image_dir.glob(f"*.{self.ext_img}"):
            img = cv2.imread(str(file))
            if img is not None:
                h, w = img.shape[:2]
                self.image_shapes[file.name] = (h, w)
                self.images.append((file.name, img))
        return self.images

    def load_bounding_boxes(self) -> Dict[str, List[BoundingBox]]:
        """
        Load bounding boxes from associated text files.
        
        Returns:
            Dict[str, List[BoundingBox]]: Dictionary mapping filenames to lists of bounding boxes
        """
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

    def get_image(self, name: str) -> Optional[Any]:
        """
        Retrieve image by filename.
        
        Args:
            name (str): Filename of the image
            
        Returns:
            Optional[Any]: The image if found, None otherwise
        """
        return next((img for fname, img in self.images if fname == name), None)

    def get_bounding_boxes(self, name: str) -> List[Union[BoundingBox, PixelBoundingBox]]:
        """
        Retrieve bounding boxes for a given image filename.
        
        Args:
            name (str): Filename of the image
            
        Returns:
            List[Union[BoundingBox, PixelBoundingBox]]: List of bounding boxes
        """
        return self.bboxes.get(name, [])

    def denormalize_bounding_boxes_(self) -> None:
        """
        Convert YOLO-normalized BoundingBox instances to absolute PixelBoundingBox and update self.bboxes.
        
        Raises:
            ValueError: If images have not been loaded
            Warning: If bounding boxes are already in pixel format
        """
        if not self.images:
            raise ValueError("Images must be loaded before denormalizing bounding boxes.")
        if self.is_pixel_format():
            raise Warning("Image already in pixel format.")

        for img_name, img in self.images:
            h, w = self.image_shapes[img_name]
            boxes = self.bboxes.get(img_name, [])
            abs_boxes = []
            for box in boxes:
                abs_boxes.append(box.to_pixel(w, h))
            self.bboxes[img_name] = abs_boxes
        self.bbox_format = "pixel"

    def normalize_bounding_boxes(self) -> None:
        """
        Convert PixelBoundingBox instances to YOLO-normalized BoundingBox format and update self.bboxes.
        
        Raises:
            ValueError: If images have not been loaded
            Warning: If bounding boxes are already in YOLO format
        """
        if not self.images:
            raise ValueError("Images must be loaded before normalizing bounding boxes.")
        if self.is_yolo_format():
            raise Warning("Image already in YOLO format.")

        for img_name, img in self.images:
            h, w = self.image_shapes[img_name]
            boxes = self.bboxes.get(img_name, [])
            norm_boxes = []
            for box in boxes:
                norm_boxes.append(box.to_normalized(w, h))
            self.bboxes[img_name] = norm_boxes
        self.bbox_format = "yolo"

    def is_yolo_format(self) -> bool:
        """
        Check if bounding boxes are in YOLO format.
        
        Returns:
            bool: True if in YOLO format, False otherwise
        """
        return self.bbox_format == "yolo"

    def is_pixel_format(self) -> bool:
        """
        Check if bounding boxes are in pixel format.
        
        Returns:
            bool: True if in pixel format, False otherwise
        """
        return self.bbox_format == "pixel"

    def __getitem__(self, idx: int) -> Tuple[str, Any, List[Union[BoundingBox, PixelBoundingBox]]]:
        """
        Get an item from the dataset by index.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Tuple[str, Any, List[Union[BoundingBox, PixelBoundingBox]]]: Tuple containing (filename, image, bounding boxes)
        """
        fname, img = self.images[idx]
        bboxes = self.get_bounding_boxes(fname)

        if self.is_yolo_format():
            assert all(isinstance(b, BoundingBox) for b in bboxes)
        elif self.is_pixel_format():
            assert all(isinstance(b, PixelBoundingBox) for b in bboxes)

        return fname, img, bboxes

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.images)


class LabelDataset:
    """
    Handles loading and storing label data.
    
    Attributes:
        label_dir (Optional[Path]): Directory containing label data
        label_data (defaultdict): Dictionary mapping labels to data
        labels (Optional[List[str]]): List of labels
        image_ds (Optional[ImageDataset]): Associated ImageDataset
    """

    def __init__(self, label_dir: Union[str, Path] = None, image_ds: ImageDataset = None):
        """
        Initialize the LabelDataset.
        
        Args:
            label_dir (Union[str, Path], optional): Directory containing label data. Defaults to None.
            image_ds (ImageDataset, optional): Associated ImageDataset. Defaults to None.
        """
        self.label_dir = Path(label_dir) if label_dir else None
        self.label_data = defaultdict(list)
        self.labels = None
        self.extractor = None
        self.image_ds = image_ds

    def load_labels(self):
        """
        Load labels from the label directory or extract from images.
        
        Raises:
            ValueError: If no label directory or image dataset is provided
            NotImplementedError: If label loading from directory is not implemented
        """
        if not self.label_dir:
            if not self.image_ds:
                raise ValueError("Either label_dir or image_ds must be provided.")
            
            from augmentv1.data.recycling_code_extractor import RecyclingCodeExtractor
            self.extractor = RecyclingCodeExtractor(self.image_ds)
            self.extractor.extract_code()
            self.labels = self.extractor.class_to_crops
        else:
            # Load labels from directory
            if (self.label_dir / "classes.txt").exists():
                with open(self.label_dir / "classes.txt", "r") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            else:
                raise NotImplementedError("No label loading from dir implemented.")
        
        return self.labels

    def get_random_label(self) -> str:
        """
        Get a random label from the loaded labels.
        
        Returns:
            str: A random label
            
        Raises:
            ValueError: If labels have not been loaded
        """
        if not self.labels:
            raise ValueError("Labels have not been loaded yet.")
        return random.choice(self.labels)

    def get_extractor(self):
        return self.extractor


class DataLoader:
    """
    Coordinates loading of image and label datasets.
    
    Attributes:
        image_dataset (Optional[ImageDataset]): Dataset for images and bounding boxes
        label_dataset (Optional[LabelDataset]): Dataset for labels
        images (Optional[List[Tuple[str, Any]]]): List of loaded images
        bboxes (Optional[Dict[str, List[Union[BoundingBox, PixelBoundingBox]]]]): Dictionary of bounding boxes
        labels (Optional[List[str]]): List of labels
    """

    def __init__(self, image_dir: Optional[str] = None, label_dir: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            image_dir (Optional[str], optional): Directory containing images. Defaults to None.
            label_dir (Optional[str], optional): Directory containing labels. Defaults to None.
        """
        self.image_dataset: Optional[ImageDataset] = ImageDataset(image_dir) if image_dir else None
        self.label_dataset: Optional[LabelDataset] = LabelDataset(label_dir) if label_dir else None

        self.images: Optional[List[Tuple[str, Any]]] = None
        self.bboxes: Optional[Dict[str, List[Union[BoundingBox, PixelBoundingBox]]]] = None
        self.labels: Optional[List[str]] = None

    def load_data(self) -> None:
        """
        Load both images and labels.
        
        Raises:
            ValueError: If image or label dataset is not initialized
        """
        if not self.image_dataset or not self.label_dataset:
            raise ValueError("Both image and label datasets must be initialized.")
        self.images = self.image_dataset.load_images()
        self.bboxes = self.image_dataset.load_bounding_boxes()
        self.labels = self.label_dataset.load_labels()

    def load_images(self) -> None:
        """
        Load only images and bounding boxes.
        
        Raises:
            ValueError: If image dataset is not initialized
        """
        if self.image_dataset is None:
            raise ValueError("ImageDataset not initialized.")
        self.images = self.image_dataset.load_images()
        self.bboxes = self.image_dataset.load_bounding_boxes()

    def load_labels(self) -> None:
        """
        Load only labels.
        
        Raises:
            ValueError: If label dataset is not initialized and image dataset is not available
        """
        if not self.label_dataset:
            try:
                self.label_dataset = LabelDataset(image_ds=self.image_dataset)
                self.label_dataset.load_labels()
                self.labels = self.label_dataset.labels
            except ValueError:
                raise ValueError("LabelDataset or Image Data not initialized.")
        else:
            self.label_dataset.load_labels()

    def get_images(self) -> Optional[List[Tuple[str, Any]]]:
        """
        Get the loaded images.
        
        Returns:
            Optional[List[Tuple[str, Any]]]: List of tuples containing (filename, image)
        """
        return self.images

    def get_labels(self) -> Optional[List[str]]:
        """
        Get the loaded labels.
        
        Returns:
            Optional[List[str]]: List of labels
        """
        return self.labels

    def query(self, by: str, value) -> tuple:
        """
        Query image or bounding box data by class or filename.

        Args:
            by (str): Either "class" or "filename".
            value: Class ID (int) or filename (str) depending on the query type.

        Returns:
            List of matching bounding boxes, image crops, or metadata.
            
        Raises:
            ValueError: If image dataset is not loaded or query type is invalid
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
            return [], None

        else:
            raise ValueError("Invalid query type. Use 'class' or 'filename'.")

    def summary(self, show_class_names: bool = False) -> None:
        """
        Print a summary of the dataset including class counts, number of files, total size, etc.

        Args:
            show_class_names (bool, optional): Whether to show class names from the LabelDataset. Defaults to False.
            
        Raises:
            ValueError: If image dataset is not loaded
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

    def plot_class_labels(self, cls: int, n: Optional[int] = None, cols: int = 5, figsize: Tuple[int, int] = (15, 8)):

        if self.label_dataset.extractor is not None:
            self.label_dataset.extractor.display_class(cls, n, cols, figsize)