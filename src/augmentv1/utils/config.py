"""
Configuration management module.

This module provides functionality to load, validate, and access configuration
settings from YAML or JSON files.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass


class Config:
    """
    Configuration manager for the application.

    This class handles loading configuration from YAML or JSON files,
    providing access to configuration values, and validating the configuration.

    Attributes:
        config_data (Dict[str, Any]): The loaded configuration data
        config_file (Path): Path to the configuration file
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the Config object.

        Args:
            config_file (Optional[Union[str, Path]], optional): Path to the configuration file.
                If None, will look for config.yaml or config.json in the current directory.
                Defaults to None.

        Raises:
            ConfigurationError: If the configuration file cannot be found or loaded
        """
        self.config_data: Dict[str, Any] = {}
        self.config_file: Optional[Path] = Path(config_file) if config_file else None

        # If no config file specified, look for default config files
        if not self.config_file:
            for default_name in ["config.yaml", "config.yml", "config.json"]:
                if os.path.exists(default_name):
                    self.config_file = Path(default_name)
                    break

        # Load configuration if file is specified or found
        if self.config_file:
            self.load_config()

    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_file (Optional[Union[str, Path]], optional): Path to the configuration file.
                If None, uses the previously specified file. Defaults to None.

        Returns:
            Dict[str, Any]: The loaded configuration data

        Raises:
            ConfigurationError: If the configuration file cannot be found or loaded
        """
        if config_file:
            self.config_file = Path(config_file)

        if not self.config_file:
            raise ConfigurationError("No configuration file specified")

        if not self.config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == '.json':
                    self.config_data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {self.config_file.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {str(e)}")

        return self.config_data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): The configuration key (can use dot notation for nested keys)
            default (Any, optional): Default value to return if key is not found. Defaults to None.

        Returns:
            Any: The configuration value or default if not found
        """
        if '.' in key:
            # Handle nested keys with dot notation
            parts = key.split('.')
            current = self.config_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        else:
            # Simple key lookup
            return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key (str): The configuration key (can use dot notation for nested keys)
            value (Any): The value to set
        """
        if '.' in key:
            # Handle nested keys with dot notation
            parts = key.split('.')
            current = self.config_data
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Simple key assignment
            self.config_data[key] = value

    def save(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a file.

        Args:
            config_file (Optional[Union[str, Path]], optional): Path to save the configuration file.
                If None, uses the previously specified file. Defaults to None.

        Raises:
            ConfigurationError: If the configuration file cannot be saved
        """
        if config_file:
            self.config_file = Path(config_file)

        if not self.config_file:
            raise ConfigurationError("No configuration file specified for saving")

        try:
            with open(self.config_file, 'w') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config_data, f, default_flow_style=False)
                elif self.config_file.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {self.config_file.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration file: {str(e)}")

    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that all required keys are present in the configuration.

        Args:
            required_keys (list): List of required keys

        Returns:
            bool: True if all required keys are present, False otherwise

        Raises:
            ConfigurationError: If any required keys are missing
        """
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            raise ConfigurationError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        return True

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary-like access.

        Args:
            key (str): The configuration key

        Returns:
            Any: The configuration value

        Raises:
            KeyError: If the key is not found
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary-like access.

        Args:
            key (str): The configuration key
            value (Any): The value to set
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key (str): The configuration key

        Returns:
            bool: True if the key exists, False otherwise
        """
        return self.get(key) is not None


# Create a default configuration instance
config = Config()


def create_default_config(config_file: Union[str, Path]) -> None:
    """
    Create a default configuration file.

    Args:
        config_file (Union[str, Path]): Path to save the configuration file
    """
    default_config = {
        "data": {
            "image_dir": "data/raw/train",
            "label_dir": "data/raw/train",
            "image_ext": "png",
            "label_ext": "txt"
        },
        "augmentation": {
            "enabled": True,
            "techniques": ["horizontal_flip", "vertical_flip", "rotation", "brightness", "contrast", "swap_labels"],
            "augmentations_per_image": 3
        },
        "model": {
            "architecture": "yolo",
            "input_size": [416, 416],
            "batch_size": 16,
            "epochs": 100
        },
        "logging": {
            "level": "INFO",
            "file": "logs/augmentv1.log"
        }
    }

    config_path = Path(config_file)

    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(default_config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(default_config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
    except Exception as e:
        raise ConfigurationError(f"Error creating default configuration file: {str(e)}")
