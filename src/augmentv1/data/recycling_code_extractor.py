"""
Recycling code extraction module.

This module provides functionality to extract recycling code regions from images
based on bounding box annotations.
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from augmentv1.data.data_loader import ImageDataset


class RecyclingCodeExtractor:
    """
    Extracts and manages recycling code regions from images.
    
    This class extracts image regions containing recycling codes based on
    bounding box annotations and provides methods to display and analyze them.
    
    Attributes:
        image_dataset (ImageDataset): Dataset containing images and bounding boxes
        class_to_crops (Dict[int, List[Tuple[str, Any]]]): Dictionary mapping class IDs to lists of crops
    """
    
    def __init__(self, image_dataset: ImageDataset):
        """
        Initialize the RecyclingCodeExtractor.
        
        Args:
            image_dataset (ImageDataset): Dataset containing images and bounding boxes
        """
        self.image_dataset = image_dataset
        self.class_to_crops: Dict[int, List[Tuple[str, Any]]] = {}
        
        # Ensure bounding boxes are in pixel format for extraction
        if image_dataset.is_yolo_format():
            image_dataset.denormalize_bounding_boxes_()

    def extract_code(self) -> Dict[int, List[Tuple[str, Any]]]:
        """
        Extract recycling code regions from images based on bounding boxes.
        
        Returns:
            Dict[int, List[Tuple[str, Any]]]: Dictionary mapping class IDs to lists of crops
        """
        for fname, img, boxes in self.image_dataset:
            for box in boxes:
                x1, y1, x2, y2, cls = box
                cropped = img[y1:y2, x1:x2]

                if cls not in self.class_to_crops:
                    self.class_to_crops[cls] = []

                self.class_to_crops[cls].append((fname, cropped))
                
        return self.class_to_crops

    def display_class(self, cls: int, n: Optional[int] = None, cols: int = 5, figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Display up to `n` cropped recycling code images belonging to a specific class in a grid.
        
        Args:
            cls (int): Class ID to display
            n (Optional[int], optional): Number of samples to display. Defaults to None (all).
            cols (int, optional): Number of columns in the display grid. Defaults to 5.
            figsize (Tuple[int, int], optional): Figure size (width, height). Defaults to (15, 8).
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
        
    def get_class_samples(self, cls: int, n: Optional[int] = None) -> List[Tuple[str, Any]]:
        """
        Get a list of image crops for a specific class.
        
        Args:
            cls (int): Class ID to retrieve
            n (Optional[int], optional): Number of samples to retrieve. Defaults to None (all).
            
        Returns:
            List[Tuple[str, Any]]: List of tuples containing (filename, crop)
        """
        items = self.class_to_crops.get(cls, [])
        
        if not items:
            return []
            
        if n is not None:
            return random.sample(items, min(n, len(items)))
        
        return items