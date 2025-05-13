"""
Bounding box classes for representing object annotations.

This module provides classes for representing bounding boxes in both
normalized (YOLO) format and pixel format.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    """
    Represents a bounding box in normalized YOLO format.
    
    Attributes:
        x (float): Normalized center x-coordinate (0-1)
        y (float): Normalized center y-coordinate (0-1)
        w (float): Normalized width (0-1)
        h (float): Normalized height (0-1)
        cls (int): Class ID
    """
    x: float
    y: float
    w: float
    h: float
    cls: int

    def __iter__(self):
        """Allow unpacking the bounding box as a tuple."""
        yield self.x
        yield self.y
        yield self.w
        yield self.h
        yield self.cls
        
    def to_pixel(self, image_width: int, image_height: int) -> 'PixelBoundingBox':
        """
        Convert normalized bounding box to pixel coordinates.
        
        Args:
            image_width (int): Width of the image
            image_height (int): Height of the image
            
        Returns:
            PixelBoundingBox: Bounding box in pixel coordinates
        """
        x1 = int((self.x - self.w / 2) * image_width)
        y1 = int((self.y - self.h / 2) * image_height)
        x2 = int((self.x + self.w / 2) * image_width)
        y2 = int((self.y + self.h / 2) * image_height)
        return PixelBoundingBox(x1, y1, x2, y2, self.cls)


@dataclass
class PixelBoundingBox:
    """
    Represents a bounding box in pixel coordinates.
    
    Attributes:
        x1 (int): Top-left x-coordinate
        y1 (int): Top-left y-coordinate
        x2 (int): Bottom-right x-coordinate
        y2 (int): Bottom-right y-coordinate
        cls (int): Class ID
    """
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int

    def __iter__(self):
        """Allow unpacking the bounding box as a tuple."""
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2
        yield self.cls
        
    def to_normalized(self, image_width: int, image_height: int) -> BoundingBox:
        """
        Convert pixel coordinates to normalized bounding box.
        
        Args:
            image_width (int): Width of the image
            image_height (int): Height of the image
            
        Returns:
            BoundingBox: Bounding box in normalized coordinates
        """
        x_center = ((self.x1 + self.x2) / 2) / image_width
        y_center = ((self.y1 + self.y2) / 2) / image_height
        box_w = (self.x2 - self.x1) / image_width
        box_h = (self.y2 - self.y1) / image_height
        return BoundingBox(x_center, y_center, box_w, box_h, self.cls)