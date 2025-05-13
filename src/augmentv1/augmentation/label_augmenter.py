"""
Label augmentation module for recycling code detection.

This module provides functionality to augment images and their corresponding
bounding box annotations to improve model training.
"""

import random
from typing import List, Tuple, Union, Any, Dict, Optional

import cv2
import numpy as np

from augmentv1.data.data_loader import DataLoader
from augmentv1.data.bounding_box import BoundingBox, PixelBoundingBox
from augmentv1.data.recycling_code_extractor import RecyclingCodeExtractor


class LabelAugmenter:
    """
    Augments images and their corresponding bounding box annotations.

    This class provides methods to apply various augmentation techniques to images
    and their bounding box annotations, such as rotation, scaling, flipping,
    color adjustments, and more.

    Attributes:
        data_loader (DataLoader): DataLoader instance containing images and annotations
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize the LabelAugmenter.

        Args:
            data_loader (DataLoader): DataLoader instance containing images and annotations
        """
        self.data_loader = data_loader
        self._recycling_code_extractor = None
        self._label_crops = {}

    def augment_labels(self, image: np.ndarray, bboxes: List[Union[BoundingBox, PixelBoundingBox]], 
                       techniques: Optional[List[str]] = None) -> Tuple[np.ndarray, List[Union[BoundingBox, PixelBoundingBox]]]:
        """
        Apply augmentation techniques to an image and its bounding boxes.

        Args:
            image (np.ndarray): Image to augment
            bboxes (List[Union[BoundingBox, PixelBoundingBox]]): Bounding boxes to augment
            techniques (Optional[List[str]], optional): List of techniques to apply. 
                If None, a random selection will be used. Defaults to None.

        Returns:
            Tuple[np.ndarray, List[Union[BoundingBox, PixelBoundingBox]]]: Augmented image and bounding boxes
        """
        # Ensure bounding boxes are in pixel format for augmentation
        is_yolo = isinstance(bboxes[0], BoundingBox) if bboxes else False

        if is_yolo:
            h, w = image.shape[:2]
            pixel_bboxes = [box.to_pixel(w, h) for box in bboxes]
        else:
            pixel_bboxes = bboxes

        # If no techniques specified, randomly select some
        if techniques is None:
            available_techniques = [
                'horizontal_flip', 'vertical_flip', 'rotation', 'brightness',
                'contrast', 'blur', 'noise', 'cutout', 'swap_labels'
            ]
            # Randomly select 1-3 techniques
            num_techniques = random.randint(1, 3)
            techniques = random.sample(available_techniques, num_techniques)

        # Apply selected techniques
        augmented_image = image.copy()
        augmented_bboxes = pixel_bboxes.copy()

        for technique in techniques:
            if technique == 'horizontal_flip':
                augmented_image, augmented_bboxes = self.horizontal_flip(augmented_image, augmented_bboxes)
            elif technique == 'vertical_flip':
                augmented_image, augmented_bboxes = self.vertical_flip(augmented_image, augmented_bboxes)
            elif technique == 'rotation':
                angle = random.uniform(-15, 15)  # Random rotation between -15 and 15 degrees
                augmented_image, augmented_bboxes = self.rotate(augmented_image, augmented_bboxes, angle)
            elif technique == 'brightness':
                factor = random.uniform(0.5, 1.5)  # Random brightness adjustment
                augmented_image = self.adjust_brightness(augmented_image, factor)
            elif technique == 'contrast':
                factor = random.uniform(0.5, 1.5)  # Random contrast adjustment
                augmented_image = self.adjust_contrast(augmented_image, factor)
            elif technique == 'blur':
                kernel_size = random.choice([3, 5, 7])  # Random blur kernel size
                augmented_image = self.apply_blur(augmented_image, kernel_size)
            elif technique == 'noise':
                augmented_image = self.add_noise(augmented_image)
            elif technique == 'cutout':
                augmented_image = self.apply_cutout(augmented_image, augmented_bboxes)
            elif technique == 'swap_labels':
                augmented_image = self.swap_packaging_labels(augmented_image, augmented_bboxes)

        # Convert back to original format if needed
        if is_yolo:
            h, w = augmented_image.shape[:2]
            augmented_bboxes = [box.to_normalized(w, h) for box in augmented_bboxes]

        return augmented_image, augmented_bboxes

    def horizontal_flip(self, image: np.ndarray, bboxes: List[PixelBoundingBox]) -> Tuple[np.ndarray, List[PixelBoundingBox]]:
        """
        Flip the image horizontally and adjust bounding boxes accordingly.

        Args:
            image (np.ndarray): Image to flip
            bboxes (List[PixelBoundingBox]): Bounding boxes to adjust

        Returns:
            Tuple[np.ndarray, List[PixelBoundingBox]]: Flipped image and adjusted bounding boxes
        """
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
        h, w = image.shape[:2]

        flipped_bboxes = []
        for box in bboxes:
            # Flip x-coordinates
            new_x1 = w - box.x2
            new_x2 = w - box.x1
            flipped_bboxes.append(PixelBoundingBox(new_x1, box.y1, new_x2, box.y2, box.cls))

        return flipped_image, flipped_bboxes

    def vertical_flip(self, image: np.ndarray, bboxes: List[PixelBoundingBox]) -> Tuple[np.ndarray, List[PixelBoundingBox]]:
        """
        Flip the image vertically and adjust bounding boxes accordingly.

        Args:
            image (np.ndarray): Image to flip
            bboxes (List[PixelBoundingBox]): Bounding boxes to adjust

        Returns:
            Tuple[np.ndarray, List[PixelBoundingBox]]: Flipped image and adjusted bounding boxes
        """
        flipped_image = cv2.flip(image, 0)  # 0 for vertical flip
        h, w = image.shape[:2]

        flipped_bboxes = []
        for box in bboxes:
            # Flip y-coordinates
            new_y1 = h - box.y2
            new_y2 = h - box.y1
            flipped_bboxes.append(PixelBoundingBox(box.x1, new_y1, box.x2, new_y2, box.cls))

        return flipped_image, flipped_bboxes

    def rotate(self, image: np.ndarray, bboxes: List[PixelBoundingBox], angle: float) -> Tuple[np.ndarray, List[PixelBoundingBox]]:
        """
        Rotate the image and adjust bounding boxes accordingly.

        Args:
            image (np.ndarray): Image to rotate
            bboxes (List[PixelBoundingBox]): Bounding boxes to adjust
            angle (float): Rotation angle in degrees

        Returns:
            Tuple[np.ndarray, List[PixelBoundingBox]]: Rotated image and adjusted bounding boxes
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate image
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

        # Rotate bounding boxes
        rotated_bboxes = []
        for box in bboxes:
            # Get the four corners of the bounding box
            corners = np.array([
                [box.x1, box.y1],
                [box.x2, box.y1],
                [box.x2, box.y2],
                [box.x1, box.y2]
            ], dtype=np.float32)

            # Rotate corners
            corners = np.hstack((corners, np.ones((4, 1))))
            corners = np.dot(M, corners.T).T

            # Get new bounding box coordinates
            new_x1 = int(np.min(corners[:, 0]))
            new_y1 = int(np.min(corners[:, 1]))
            new_x2 = int(np.max(corners[:, 0]))
            new_y2 = int(np.max(corners[:, 1]))

            # Ensure coordinates are within image bounds
            new_x1 = max(0, min(new_x1, new_w - 1))
            new_y1 = max(0, min(new_y1, new_h - 1))
            new_x2 = max(0, min(new_x2, new_w - 1))
            new_y2 = max(0, min(new_y2, new_h - 1))

            # Add rotated bounding box
            rotated_bboxes.append(PixelBoundingBox(new_x1, new_y1, new_x2, new_y2, box.cls))

        return rotated_image, rotated_bboxes

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust the brightness of an image.

        Args:
            image (np.ndarray): Image to adjust
            factor (float): Brightness adjustment factor (>1 for brighter, <1 for darker)

        Returns:
            np.ndarray: Brightness-adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust the contrast of an image.

        Args:
            image (np.ndarray): Image to adjust
            factor (float): Contrast adjustment factor (>1 for more contrast, <1 for less contrast)

        Returns:
            np.ndarray: Contrast-adjusted image
        """
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to an image.

        Args:
            image (np.ndarray): Image to blur
            kernel_size (int, optional): Size of the blur kernel. Defaults to 5.

        Returns:
            np.ndarray: Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def add_noise(self, image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to an image.

        Args:
            image (np.ndarray): Image to add noise to
            mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
            std (float, optional): Standard deviation of the Gaussian noise. Defaults to 25.

        Returns:
            np.ndarray: Noisy image
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_cutout(self, image: np.ndarray, bboxes: List[PixelBoundingBox], n_holes: int = 1, 
                    max_h_size: int = 50, max_w_size: int = 50) -> np.ndarray:
        """
        Apply cutout augmentation to an image (randomly mask out rectangular regions).

        Args:
            image (np.ndarray): Image to apply cutout to
            bboxes (List[PixelBoundingBox]): Bounding boxes (to avoid cutting out important regions)
            n_holes (int, optional): Number of holes to cut out. Defaults to 1.
            max_h_size (int, optional): Maximum height of each hole. Defaults to 50.
            max_w_size (int, optional): Maximum width of each hole. Defaults to 50.

        Returns:
            np.ndarray: Image with cutout applied
        """
        h, w = image.shape[:2]
        result = image.copy()

        for _ in range(n_holes):
            # Random cutout size
            cutout_h = random.randint(10, max_h_size)
            cutout_w = random.randint(10, max_w_size)

            # Random cutout position
            y = random.randint(0, h - cutout_h)
            x = random.randint(0, w - cutout_w)

            # Check if cutout overlaps with any bounding box by more than 50%
            overlap_too_much = False
            for box in bboxes:
                # Calculate intersection area
                x_left = max(box.x1, x)
                y_top = max(box.y1, y)
                x_right = min(box.x2, x + cutout_w)
                y_bottom = min(box.y2, y + cutout_h)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    box_area = (box.x2 - box.x1) * (box.y2 - box.y1)

                    if intersection_area / box_area > 0.5:
                        overlap_too_much = True
                        break

            # If no significant overlap, apply cutout
            if not overlap_too_much:
                result[y:y+cutout_h, x:x+cutout_w] = 0

        return result

    def _initialize_extractor(self) -> None:
        """
        Initialize the RecyclingCodeExtractor if it hasn't been initialized yet.
        This method also caches the extracted label crops for efficient access.
        """
        if self._recycling_code_extractor is None:
            # Ensure images are loaded
            if not self.data_loader.images:
                self.data_loader.load_images()

            # Create extractor
            self._recycling_code_extractor = RecyclingCodeExtractor(self.data_loader.image_dataset)
            self._recycling_code_extractor.extract_code()
            self._label_crops = self._recycling_code_extractor.class_to_crops

    def swap_packaging_labels(self, image: np.ndarray, bboxes: List[PixelBoundingBox]) -> np.ndarray:
        """
        Replace packaging labels in an image with labels from a list of packaging labels.

        This method:
        1. Removes existing package labels (based on bounding boxes)
        2. Replaces them with packaging labels from a list
        3. Ensures the new labels have the same dimensions as the original
        4. Makes sure they blend with the background

        Args:
            image (np.ndarray): Image to augment
            bboxes (List[PixelBoundingBox]): Bounding boxes of packaging labels

        Returns:
            np.ndarray: Image with swapped packaging labels
        """
        # Initialize extractor if needed
        self._initialize_extractor()

        # If no label crops available, return original image
        if not self._label_crops:
            return image

        # Get all available class IDs
        available_classes = list(self._label_crops.keys())
        if not available_classes:
            return image

        # Create a copy of the image to modify
        result = image.copy()

        # Add alpha channel to the image if it doesn't have one
        if result.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

        # Process each bounding box
        for box in bboxes:
            # Get the dimensions of the original label
            label_width = box.x2 - box.x1
            label_height = box.y2 - box.y1

            # Skip very small labels
            if label_width < 10 or label_height < 10:
                continue

            # Extract the region coordinates
            y1, y2 = box.y1, box.y2
            x1, x2 = box.x1, box.x2

            # Ensure coordinates are within image bounds
            y1 = max(0, min(y1, result.shape[0] - 1))
            y2 = max(0, min(y2, result.shape[0]))
            x1 = max(0, min(x1, result.shape[1] - 1))
            x2 = max(0, min(x2, result.shape[1]))

            # Remove the previous icon by setting alpha channel to 0
            result[y1:y2, x1:x2, 3] = 0

            # Decide whether to use the same class or a different one
            use_same_class = random.random() < 0.5

            if use_same_class and box.cls in self._label_crops:
                # Use a label from the same class
                target_class = box.cls
            else:
                # Use a label from a random different class
                target_class = random.choice(available_classes)

            # Get a random label crop from the target class
            if target_class in self._label_crops and self._label_crops[target_class]:
                _, new_label = random.choice(self._label_crops[target_class])

                # Resize the new label to match the dimensions of the original
                new_label_resized = cv2.resize(new_label, (x2 - x1, y2 - y1))

                # Add alpha channel to the new label if it doesn't have one
                if new_label_resized.shape[2] == 3:
                    new_label_resized = cv2.cvtColor(new_label_resized, cv2.COLOR_BGR2BGRA)

                # Create a mask from the new label's non-zero pixels
                # This ensures only the actual icon is placed, not the white background
                mask = new_label_resized[:, :, 0:3].mean(axis=2) > 240  # Threshold for white pixels

                # Set alpha channel based on the mask
                new_label_resized[:, :, 3] = np.where(mask, 0, 255)

                # Place the new label in the image
                for c in range(4):  # Copy all channels including alpha
                    result[y1:y2, x1:x2, c] = np.where(
                        new_label_resized[:, :, 3] > 0,  # Where the new label has non-zero alpha
                        new_label_resized[:, :, c],      # Use the new label
                        result[y1:y2, x1:x2, c]          # Keep the original (transparent) background
                    )

        # Convert back to BGR if the input was BGR
        if image.shape[2] == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        return result

    def augment_dataset(self, augmentations_per_image: int = 3, techniques: Optional[List[str]] = None) -> Dict[str, List[Tuple[np.ndarray, List[Union[BoundingBox, PixelBoundingBox]]]]]:
        """
        Apply augmentation to the entire dataset.

        Args:
            augmentations_per_image (int, optional): Number of augmented versions to create per image. Defaults to 3.
            techniques (Optional[List[str]], optional): List of techniques to apply. If None, a random selection will be used. Defaults to None.

        Returns:
            Dict[str, List[Tuple[np.ndarray, List[Union[BoundingBox, PixelBoundingBox]]]]]: Dictionary mapping filenames to lists of augmented images and bounding boxes
        """
        if not self.data_loader.images:
            self.data_loader.load_images()

        augmented_dataset = {}

        for fname, img, bboxes in self.data_loader.image_dataset:
            augmented_dataset[fname] = []

            for _ in range(augmentations_per_image):
                augmented_img, augmented_bboxes = self.augment_labels(img, bboxes, techniques)
                augmented_dataset[fname].append((augmented_img, augmented_bboxes))

        return augmented_dataset
