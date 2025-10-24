#!/usr/bin/env python3
"""
Dataset loaders for various data formats.

Supports loading image-text pairs from CSV, JSON, directories, and HuggingFace datasets.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load image-text pairs from various data formats."""

    @staticmethod
    def from_csv(path: Path, image_col: str = "image_path", caption_col: str = "caption") -> list[tuple[Path, str]]:
        """Load image-text pairs from CSV file.

        Args:
            path: Path to CSV file
            image_col: Name of column containing image paths
            caption_col: Name of column containing captions

        Returns:
            List of (image_path, caption) tuples

        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns not found
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info(f"Loading dataset from CSV: {path}")

        pairs: list[tuple[Path, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check for required columns
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no headers: {path}")

            if image_col not in reader.fieldnames:
                raise ValueError(f"Image column '{image_col}' not found in CSV. Available: {reader.fieldnames}")

            if caption_col not in reader.fieldnames:
                raise ValueError(f"Caption column '{caption_col}' not found in CSV. Available: {reader.fieldnames}")

            for row in reader:
                image_path = Path(row[image_col])
                caption = row[caption_col]

                # Make image path absolute if relative
                if not image_path.is_absolute():
                    image_path = path.parent / image_path

                pairs.append((image_path, caption))

        logger.info(f"Loaded {len(pairs)} image-text pairs from CSV")
        return pairs

    @staticmethod
    def from_json(path: Path) -> list[tuple[Path, str]]:
        """Load image-text pairs from JSON file.

        Expected JSON format:
        [
            {"image": "path/to/image.jpg", "caption": "description"},
            ...
        ]

        or:

        {
            "data": [
                {"image": "path/to/image.jpg", "caption": "description"},
                ...
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            List of (image_path, caption) tuples

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If JSON format is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        logger.info(f"Loading dataset from JSON: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        else:
            raise ValueError(f"Unsupported JSON format. Expected list or dict with 'data' key: {path}")

        pairs: list[tuple[Path, str]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid item format. Expected dict, got {type(item)}: {item}")

            if "image" not in item:
                raise ValueError(f"Missing 'image' key in item: {item}")
            if "caption" not in item:
                raise ValueError(f"Missing 'caption' key in item: {item}")

            image_path = Path(item["image"])
            caption = item["caption"]

            # Make image path absolute if relative
            if not image_path.is_absolute():
                image_path = path.parent / image_path

            pairs.append((image_path, caption))

        logger.info(f"Loaded {len(pairs)} image-text pairs from JSON")
        return pairs

    @staticmethod
    def from_directory(
        path: Path,
        caption_file: str = "captions.txt",
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[tuple[Path, str]]:
        """Load image-text pairs from directory structure.

        Expected directory structure:
        ```
        path/
            image1.jpg
            image2.jpg
            captions.txt  # Format: "image1.jpg<tab>caption"
        ```

        or:

        ```
        path/
            image1.jpg
            image1.txt  # Caption file with same name as image
            image2.jpg
            image2.txt
        ```

        Args:
            path: Path to directory containing images
            caption_file: Name of global caption file (optional)
            extensions: Tuple of valid image extensions

        Returns:
            List of (image_path, caption) tuples

        Raises:
            FileNotFoundError: If directory not found
            ValueError: If no images found or captions missing
        """
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        logger.info(f"Loading dataset from directory: {path}")

        pairs: list[tuple[Path, str]] = []

        # Try to load from global caption file first
        caption_file_path = path / caption_file
        if caption_file_path.exists():
            logger.info(f"Loading captions from {caption_file}")
            caption_map: dict[str, str] = {}

            with open(caption_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Split on tab or whitespace
                    parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
                    if len(parts) != 2:
                        logger.warning(f"Invalid caption line: {line}")
                        continue

                    image_name, caption = parts
                    caption_map[image_name] = caption

            # Load images with captions from map
            for image_path in path.glob("*"):
                if image_path.suffix.lower() in extensions:
                    if image_path.name in caption_map:
                        pairs.append((image_path, caption_map[image_path.name]))
                    else:
                        logger.warning(f"No caption found for image: {image_path.name}")

        else:
            # Try to load from individual caption files
            logger.info("No global caption file found, looking for individual caption files")

            for image_path in path.glob("*"):
                if image_path.suffix.lower() in extensions:
                    # Look for corresponding caption file
                    caption_path = image_path.with_suffix(".txt")
                    if caption_path.exists():
                        with open(caption_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                        pairs.append((image_path, caption))
                    else:
                        logger.warning(f"No caption file found for image: {image_path.name}")

        if not pairs:
            raise ValueError(f"No image-text pairs found in directory: {path}")

        logger.info(f"Loaded {len(pairs)} image-text pairs from directory")
        return pairs

    @staticmethod
    def from_huggingface(
        dataset_name: str,
        split: str = "train",
        image_col: str = "image",
        caption_col: str = "caption",
        cache_dir: Path | None = None,
    ) -> list[tuple[Path, str]]:
        """Load image-text pairs from HuggingFace datasets.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "nlphuji/flickr30k")
            split: Dataset split to load (train, validation, test)
            image_col: Name of column containing images
            caption_col: Name of column containing captions
            cache_dir: Optional cache directory for downloaded datasets

        Returns:
            List of (image_path, caption) tuples

        Raises:
            ImportError: If datasets library not available
            ValueError: If dataset cannot be loaded or columns not found

        Note:
            This requires the `datasets` library to be installed.
            Images are saved to a temporary directory.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library not available. "
                "Install with: pip install datasets"
            )

        logger.info(f"Loading dataset from HuggingFace: {dataset_name} ({split})")

        # Load dataset
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_name}': {e}")

        # Check for required columns
        if image_col not in dataset.column_names:
            raise ValueError(
                f"Image column '{image_col}' not found in dataset. "
                f"Available: {dataset.column_names}"
            )

        if caption_col not in dataset.column_names:
            raise ValueError(
                f"Caption column '{caption_col}' not found in dataset. "
                f"Available: {dataset.column_names}"
            )

        # Create temporary directory for images
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="clip_hf_"))
        logger.info(f"Saving images to temporary directory: {temp_dir}")

        pairs: list[tuple[Path, str]] = []
        for idx, item in enumerate(dataset):
            image = item[image_col]
            caption = item[caption_col]

            # Save image to temporary directory
            image_path = temp_dir / f"image_{idx:06d}.jpg"
            image.save(image_path)

            pairs.append((image_path, caption))

        logger.info(f"Loaded {len(pairs)} image-text pairs from HuggingFace")
        return pairs
