#!/usr/bin/env python3
"""Tests for dataset loaders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data.loaders import DatasetLoader


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_from_csv(self, fixtures_dir):
        """Test loading from CSV file."""
        csv_path = fixtures_dir / "test_data.csv"
        pairs = DatasetLoader.from_csv(csv_path)

        assert len(pairs) == 5
        assert all(isinstance(img_path, Path) for img_path, _ in pairs)
        assert all(isinstance(caption, str) for _, caption in pairs)

        # Check first pair
        img_path, caption = pairs[0]
        assert img_path.name == "test_image_0.jpg"
        assert caption == "A cat sitting on a mat"

    def test_from_csv_with_custom_columns(self, temp_dir):
        """Test loading from CSV with custom column names."""
        csv_path = temp_dir / "custom.csv"
        with open(csv_path, "w") as f:
            f.write("img,text\n")
            f.write("image1.jpg,Caption 1\n")
            f.write("image2.jpg,Caption 2\n")

        pairs = DatasetLoader.from_csv(csv_path, image_col="img", caption_col="text")
        assert len(pairs) == 2
        assert pairs[0][1] == "Caption 1"

    def test_from_csv_missing_file(self):
        """Test loading from non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.from_csv(Path("nonexistent.csv"))

    def test_from_csv_missing_columns(self, temp_dir):
        """Test loading from CSV with missing columns."""
        csv_path = temp_dir / "invalid.csv"
        with open(csv_path, "w") as f:
            f.write("wrong_col1,wrong_col2\n")
            f.write("val1,val2\n")

        with pytest.raises(ValueError, match="Image column"):
            DatasetLoader.from_csv(csv_path)

    def test_from_json(self, fixtures_dir):
        """Test loading from JSON file."""
        json_path = fixtures_dir / "test_captions.json"
        pairs = DatasetLoader.from_json(json_path)

        assert len(pairs) == 5
        assert all(isinstance(img_path, Path) for img_path, _ in pairs)
        assert all(isinstance(caption, str) for _, caption in pairs)

        # Check first pair
        img_path, caption = pairs[0]
        assert img_path.name == "test_image_0.jpg"
        assert caption == "A cat sitting on a mat"

    def test_from_json_with_data_key(self, temp_dir):
        """Test loading from JSON with 'data' key."""
        json_path = temp_dir / "wrapped.json"
        data = {
            "data": [
                {"image": "image1.jpg", "caption": "Caption 1"},
                {"image": "image2.jpg", "caption": "Caption 2"},
            ]
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        pairs = DatasetLoader.from_json(json_path)
        assert len(pairs) == 2

    def test_from_json_missing_file(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.from_json(Path("nonexistent.json"))

    def test_from_json_invalid_format(self, temp_dir):
        """Test loading from JSON with invalid format."""
        json_path = temp_dir / "invalid.json"
        with open(json_path, "w") as f:
            json.dump({"wrong_key": []}, f)

        with pytest.raises(ValueError, match="Unsupported JSON format"):
            DatasetLoader.from_json(json_path)

    def test_from_json_missing_keys(self, temp_dir):
        """Test loading from JSON with missing keys."""
        json_path = temp_dir / "missing_keys.json"
        data = [{"image": "image1.jpg"}]  # Missing caption
        with open(json_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Missing 'caption' key"):
            DatasetLoader.from_json(json_path)

    def test_from_directory_with_caption_file(self, temp_dir, fixtures_dir):
        """Test loading from directory with global caption file."""
        # Copy test images to temp directory
        for i in range(3):
            src = fixtures_dir / f"test_image_{i}.jpg"
            dst = temp_dir / f"test_image_{i}.jpg"
            import shutil
            shutil.copy(src, dst)

        # Create caption file
        caption_file = temp_dir / "captions.txt"
        with open(caption_file, "w") as f:
            f.write("test_image_0.jpg\tCaption 0\n")
            f.write("test_image_1.jpg\tCaption 1\n")
            f.write("test_image_2.jpg\tCaption 2\n")

        pairs = DatasetLoader.from_directory(temp_dir)
        assert len(pairs) == 3

    def test_from_directory_with_individual_captions(self, temp_dir, fixtures_dir):
        """Test loading from directory with individual caption files."""
        # Copy test images to temp directory
        for i in range(3):
            src = fixtures_dir / f"test_image_{i}.jpg"
            dst = temp_dir / f"test_image_{i}.jpg"
            import shutil
            shutil.copy(src, dst)

            # Create corresponding caption file
            caption_file = temp_dir / f"test_image_{i}.txt"
            with open(caption_file, "w") as f:
                f.write(f"Caption {i}")

        pairs = DatasetLoader.from_directory(temp_dir)
        assert len(pairs) == 3

        # Check that captions are correctly associated (order may vary due to glob)
        captions = {pair[0].name: pair[1] for pair in pairs}
        assert captions["test_image_0.jpg"] == "Caption 0"
        assert captions["test_image_1.jpg"] == "Caption 1"
        assert captions["test_image_2.jpg"] == "Caption 2"

    def test_from_directory_missing_dir(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.from_directory(Path("nonexistent"))

    def test_from_directory_not_a_dir(self, temp_dir):
        """Test loading from file instead of directory."""
        file_path = temp_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="not a directory"):
            DatasetLoader.from_directory(file_path)

    def test_from_directory_no_images(self, temp_dir):
        """Test loading from directory with no images."""
        caption_file = temp_dir / "captions.txt"
        with open(caption_file, "w") as f:
            f.write("image.jpg\tCaption\n")

        with pytest.raises(ValueError, match="No image-text pairs found"):
            DatasetLoader.from_directory(temp_dir)

    @pytest.mark.skip(reason="Requires HuggingFace datasets library")
    def test_from_huggingface(self):
        """Test loading from HuggingFace datasets.

        This test is skipped by default as it requires internet connection
        and the datasets library.
        """
        pairs = DatasetLoader.from_huggingface(
            "nlphuji/flickr30k",
            split="test[:5]",  # Load only 5 samples
        )
        assert len(pairs) == 5
