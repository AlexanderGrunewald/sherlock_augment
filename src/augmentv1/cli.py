"""
Command-line interface for the augmentv1 package.

This module provides a command-line interface for common operations such as
creating configuration files, loading and displaying dataset summaries,
and augmenting datasets.
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
from typing import List, Optional

from augmentv1 import __version__
from augmentv1.utils.config import create_default_config, Config, ConfigurationError
from augmentv1.utils.logging import setup_logging, info, error
from augmentv1.data.data_loader import DataLoader
from augmentv1.augmentation.label_augmenter import LabelAugmenter


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Returns:
        argparse.ArgumentParser: The argument parser
    """
    parser = argparse.ArgumentParser(
        description="AugmentV1: Recycling Code Detection and Data Augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version", action="version", version=f"AugmentV1 {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Configuration management"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config command to execute")

    # Config create command
    config_create_parser = config_subparsers.add_parser(
        "create", help="Create a default configuration file"
    )
    config_create_parser.add_argument(
        "--output", "-o", type=str, default="config.yaml",
        help="Output file path"
    )

    # Data command
    data_parser = subparsers.add_parser(
        "data", help="Data management"
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data command to execute")

    # Data summary command
    data_summary_parser = data_subparsers.add_parser(
        "summary", help="Display dataset summary"
    )
    data_summary_parser.add_argument(
        "--image-dir", "-i", type=str, required=True,
        help="Directory containing images and annotations"
    )
    data_summary_parser.add_argument(
        "--show-class-names", "-c", action="store_true",
        help="Show class names in summary"
    )

    # Augment command
    augment_parser = subparsers.add_parser(
        "augment", help="Augment a dataset"
    )
    augment_parser.add_argument(
        "--image-dir", "-i", type=str, required=True,
        help="Directory containing images and annotations"
    )
    augment_parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Directory to save augmented images and annotations"
    )
    augment_parser.add_argument(
        "--config", "-c", type=str,
        help="Configuration file path"
    )
    augment_parser.add_argument(
        "--augmentations", "-a", type=int, default=3,
        help="Number of augmentations per image"
    )
    augment_parser.add_argument(
        "--techniques", "-t", type=str, nargs="+",
        choices=["horizontal_flip", "vertical_flip", "rotation", "brightness", "contrast", "blur", "noise", "cutout", "swap_labels"],
        help="Augmentation techniques to use"
    )

    return parser


def config_create(args: argparse.Namespace) -> int:
    """
    Create a default configuration file.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        create_default_config(args.output)
        info(f"Created default configuration file: {args.output}")
        return 0
    except ConfigurationError as e:
        error(f"Error creating configuration file: {str(e)}")
        return 1


def data_summary(args: argparse.Namespace) -> int:
    """
    Display dataset summary.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        data_loader = DataLoader(image_dir=args.image_dir)
        data_loader.load_images()
        data_loader.summary(show_class_names=args.show_class_names)
        return 0
    except Exception as e:
        error(f"Error displaying dataset summary: {str(e)}")
        return 1


def augment_dataset(args: argparse.Namespace) -> int:
    """
    Augment a dataset.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load configuration if specified
        config = None
        if args.config:
            config = Config(args.config)

        # Load dataset
        data_loader = DataLoader(image_dir=args.image_dir)
        data_loader.load_images()

        # Create augmenter
        augmenter = LabelAugmenter(data_loader)

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Augment dataset
        info(f"Augmenting dataset with {args.augmentations} augmentations per image...")
        augmented_dataset = augmenter.augment_dataset(augmentations_per_image=args.augmentations, techniques=args.techniques)

        # Save augmented images and annotations
        info(f"Saving augmented dataset to {args.output_dir}...")
        num_saved = 0
        for fname, augmented_items in augmented_dataset.items():
            for i, (augmented_img, augmented_bboxes) in enumerate(augmented_items):
                # Generate output filename
                base_name = os.path.splitext(fname)[0]
                output_img_path = os.path.join(args.output_dir, f"{base_name}_aug{i+1}.png")
                output_txt_path = os.path.join(args.output_dir, f"{base_name}_aug{i+1}.txt")

                # Save augmented image
                cv2.imwrite(output_img_path, augmented_img)

                # Save augmented annotations
                with open(output_txt_path, "w") as f:
                    for box in augmented_bboxes:
                        if hasattr(box, "x"):  # BoundingBox (YOLO format)
                            f.write(f"{box.cls} {box.x} {box.y} {box.w} {box.h}\n")
                        else:  # PixelBoundingBox
                            # Convert to YOLO format for saving
                            h, w = augmented_img.shape[:2]
                            yolo_box = box.to_normalized(w, h)
                            f.write(f"{yolo_box.cls} {yolo_box.x} {yolo_box.y} {yolo_box.w} {yolo_box.h}\n")

                num_saved += 1

        info(f"Successfully saved {num_saved} augmented images and annotations.")
        return 0
    except Exception as e:
        error(f"Error augmenting dataset: {str(e)}")
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the command-line interface.

    Args:
        args (Optional[List[str]], optional): Command-line arguments. Defaults to None.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Set up logging
    setup_logging()

    # Parse arguments
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Execute command
    if parsed_args.command == "config":
        if parsed_args.config_command == "create":
            return config_create(parsed_args)
        else:
            parser.print_help()
            return 1
    elif parsed_args.command == "data":
        if parsed_args.data_command == "summary":
            return data_summary(parsed_args)
        else:
            parser.print_help()
            return 1
    elif parsed_args.command == "augment":
        return augment_dataset(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
