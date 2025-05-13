"""
Logging module for the application.

This module provides a centralized logging system that can be configured
to output logs to files, console, etc.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

from augmentv1.utils.config import config


class LoggingError(Exception):
    """Exception raised for errors in the logging system."""
    pass


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up the logging system.
    
    Args:
        level (str, optional): Logging level. Defaults to "INFO".
        log_file (Optional[str], optional): Path to the log file. Defaults to None.
        console (bool, optional): Whether to log to console. Defaults to True.
        log_format (Optional[str], optional): Log message format. Defaults to None.
        date_format (Optional[str], optional): Date format for log messages. Defaults to None.
        
    Returns:
        logging.Logger: The configured logger
        
    Raises:
        LoggingError: If the logging system cannot be set up
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set the logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Set default formats if not provided
    if not log_format:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if not date_format:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    formatter = logging.Formatter(log_format, date_format)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            log_path = Path(log_file)
            os.makedirs(log_path.parent, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            raise LoggingError(f"Error setting up log file: {str(e)}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: The logger
    """
    return logging.getLogger(name)


# Set up logging from configuration
def configure_logging_from_config() -> logging.Logger:
    """
    Configure logging from the application configuration.
    
    Returns:
        logging.Logger: The configured logger
    """
    log_level = config.get("logging.level", "INFO")
    log_file = config.get("logging.file")
    console = config.get("logging.console", True)
    log_format = config.get("logging.format")
    date_format = config.get("logging.date_format")
    
    return setup_logging(
        level=log_level,
        log_file=log_file,
        console=console,
        log_format=log_format,
        date_format=date_format
    )


# Initialize logging if configuration is available
try:
    logger = configure_logging_from_config()
except Exception as e:
    # Fall back to basic logging if configuration fails
    logger = setup_logging()
    logger.warning(f"Failed to configure logging from config: {str(e)}")


# Convenience functions for logging
def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    logger.critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """Log an exception message."""
    logger.exception(msg, *args, **kwargs)