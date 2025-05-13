"""
Setup script for the augmentv1 package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="augmentv1",
    version="0.1.0",
    description="A package for recycling code detection and data augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Grunewald",
    author_email="alexgrunewald123@gmail.com",
    url="https://github.com/AlexanderGrunewald/sherlock_augment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "humanize>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "augmentv1=augmentv1.cli:main",
        ],
    },
)