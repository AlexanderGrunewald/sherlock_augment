"""
This package contains deprecated functionality that will be removed in future versions.
Please refer to the current implementation in the main package.
"""

__version__ = '1.0.0'

import warnings

warnings.warn(
    "The deprecated package is scheduled for removal in future versions. "
    "Please update your code to use the current implementation.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []  # Empty list as all functionality is deprecated
