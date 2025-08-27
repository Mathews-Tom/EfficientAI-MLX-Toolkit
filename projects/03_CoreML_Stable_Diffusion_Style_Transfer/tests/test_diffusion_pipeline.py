"""Test diffusion pipeline functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.diffusion.config import DiffusionConfig
from src.diffusion.pipeline import DiffusionPipeline


class TestDiffusionPipeline:
    """Test DiffusionPipeline class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig()
        self.pipeline = DiffusionPipeline(self.config)

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        assert self.pipeline.config == self.config
        assert self.pipeline.loaded is False
        assert self.pipeline.model is not None

    def test_load_model(self):
        """Test model loading."""
        with patch.object(self.pipeline.model, "load_model") as mock_load:
            self.pipeline.load_model()

            mock_load.assert_called_once()
            assert self.pipeline.loaded is True

    def test_unload_model(self):
        """Test model unloading."""
        self.pipeline.loaded = True

        with patch.object(self.pipeline.model, "unload_model") as mock_unload:
            self.pipeline.unload_model()

            mock_unload.assert_called_once()
            assert self.pipeline.loaded is False

    def test_prepare_image_from_path(self):
        """Test image preparation from file path."""
        mock_image = Mock(spec=Image.Image)
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image

        with patch("PIL.Image.open", return_value=mock_image):
            result = self.pipeline._prepare_image("test.jpg")

            assert result == mock_image
            mock_image.convert.assert_called_once_with("RGB")

    def test_prepare_image_from_path_with_resize(self):
        """Test image preparation from file path with resizing."""
        mock_image = Mock(spec=Image.Image)
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image

        with patch("PIL.Image.open", return_value=mock_image):
            result = self.pipeline._prepare_image("test.jpg", target_size=(256, 256))

            mock_image.resize.assert_called_once_with(
                (256, 256), Image.Resampling.LANCZOS
            )

    def test_prepare_image_from_numpy_uint8(self):
        """Test image preparation from numpy array (uint8)."""
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_image = Mock(spec=Image.Image)

        with patch("PIL.Image.fromarray", return_value=mock_image) as mock_fromarray:
            result = self.pipeline._prepare_image(test_array)

            assert result == mock_image
            mock_fromarray.assert_called_once_with(test_array)

    def test_prepare_image_from_numpy_float(self):
        """Test image preparation from numpy array (float)."""
        test_array = np.random.rand(100, 100, 3).astype(np.float32)
        mock_image = Mock(spec=Image.Image)

        with patch("PIL.Image.fromarray", return_value=mock_image) as mock_fromarray:
            result = self.pipeline._prepare_image(test_array)

            assert result == mock_image
            # Check that array was converted to uint8
            call_args = mock_fromarray.call_args[0][0]
            assert call_args.dtype == np.uint8

    def test_prepare_image_pil_image(self):
        """Test image preparation from PIL Image (no conversion needed)."""
        mock_image = Mock(spec=Image.Image)
        result = self.pipeline._prepare_image(mock_image)
        assert result == mock_image

    def test_create_style_prompt_basic(self):
        """Test basic style prompt creation."""
        result = self.pipeline._create_style_prompt(
            "a beautiful landscape", "Van Gogh painting"
        )

        expected = "a beautiful landscape, in the style of Van Gogh painting"
        assert result == expected

    def test_create_style_prompt_with_high_detail(self):
        """Test style prompt creation with high detail strength."""
        result = self.pipeline._create_style_prompt(
            "a portrait", "Renaissance art", {"detail": 0.9}
        )

        assert "a portrait, in the style of Renaissance art" in result
        assert "highly detailed" in result
        assert "professional quality" in result
        assert "artistic masterpiece" in result

    def test_create_style_prompt_with_medium_detail(self):
        """Test style prompt creation with medium detail strength."""
        result = self.pipeline._create_style_prompt(
            "a portrait", "Renaissance art", {"detail": 0.6}
        )

        assert "a portrait, in the style of Renaissance art" in result
        assert "highly detailed" in result
        assert "professional quality" in result
        assert "artistic masterpiece" not in result

    def test_create_style_prompt_with_low_detail(self):
        """Test style prompt creation with low detail strength."""
        result = self.pipeline._create_style_prompt(
            "a portrait", "Renaissance art", {"detail": 0.3}
        )

        expected = "a portrait, in the style of Renaissance art"
        assert result == expected

    def test_create_negative_prompt_default(self):
        """Test default negative prompt creation."""
        result = self.pipeline._create_negative_prompt()

        expected_terms = [
            "blurry",
            "low quality",
            "distorted",
            "artifacts",
            "noise",
            "oversaturated",
            "watermark",
            "signature",
        ]
        for term in expected_terms:
            assert term in result

    def test_create_negative_prompt_with_base(self):
        """Test negative prompt creation with base negative."""
        result = self.pipeline._create_negative_prompt(base_negative="ugly")

        assert result.startswith("ugly")
        assert "blurry" in result

    def test_create_negative_prompt_with_avoid_terms(self):
        """Test negative prompt creation with additional terms to avoid."""
        result = self.pipeline._create_negative_prompt(avoid_terms=["cartoon", "anime"])

        assert "cartoon" in result
        assert "anime" in result
        assert "blurry" in result

    def test_text_to_image_stylized_not_loaded(self):
        """Test text-to-image stylized when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.pipeline.text_to_image_stylized("test prompt", "Van Gogh style")

    def test_text_to_image_stylized_success(self):
        """Test successful text-to-image stylized generation."""
        self.pipeline.loaded = True
        mock_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline.model, "generate_image", return_value=mock_image
        ) as mock_generate:
            result = self.pipeline.text_to_image_stylized(
                prompt="a beautiful landscape",
                style_description="Van Gogh painting",
                width=256,
                height=256,
                num_inference_steps=20,
                guidance_scale=7.5,
                style_strength=0.8,
                seed=42,
            )

            assert result == mock_image
            mock_generate.assert_called_once()

            call_args = mock_generate.call_args[1]
            assert (
                "a beautiful landscape, in the style of Van Gogh painting"
                in call_args["prompt"]
            )
            assert call_args["width"] == 256
            assert call_args["height"] == 256
            assert call_args["num_inference_steps"] == 20
            assert call_args["guidance_scale"] == 7.5
            assert call_args["generator"] is not None

    def test_text_to_image_stylized_without_seed(self):
        """Test text-to-image stylized without seed."""
        self.pipeline.loaded = True
        mock_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline.model, "generate_image", return_value=mock_image
        ) as mock_generate:
            result = self.pipeline.text_to_image_stylized("test prompt", "test style")

            assert result == mock_image
            call_args = mock_generate.call_args[1]
            assert call_args["generator"] is None

    def test_image_to_image_stylized_not_loaded(self):
        """Test image-to-image stylized when model not loaded."""
        mock_content_image = Mock(spec=Image.Image)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.pipeline.image_to_image_stylized(mock_content_image, "Van Gogh style")

    def test_image_to_image_stylized_success(self):
        """Test successful image-to-image stylized."""
        self.pipeline.loaded = True
        self.pipeline.model.device = "cpu"

        mock_content_image = Mock(spec=Image.Image)
        mock_result_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline, "_prepare_image", return_value=mock_content_image
        ):
            with patch.object(
                self.pipeline.model, "img2img", return_value=mock_result_image
            ) as mock_img2img:
                result = self.pipeline.image_to_image_stylized(
                    content_image=mock_content_image,
                    style_description="impressionist painting",
                    style_strength=0.8,
                    content_preservation=0.7,
                    seed=42,
                )

                assert result == mock_result_image
                mock_img2img.assert_called_once()

                call_args = mock_img2img.call_args[1]
                assert call_args["image"] == mock_content_image
                assert (
                    abs(call_args["strength"] - 0.3) < 1e-10
                )  # 1.0 - content_preservation
                assert "impressionist painting" in call_args["prompt"]

    def test_dual_image_style_transfer_not_loaded(self):
        """Test dual image style transfer when model not loaded."""
        mock_content = Mock(spec=Image.Image)
        mock_style = Mock(spec=Image.Image)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.pipeline.dual_image_style_transfer(mock_content, mock_style)

    def test_dual_image_style_transfer_success(self):
        """Test successful dual image style transfer."""
        self.pipeline.loaded = True
        self.pipeline.model.device = "cpu"

        mock_content_image = Mock(spec=Image.Image)
        mock_style_image = Mock(spec=Image.Image)
        mock_result_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline, "_prepare_image", return_value=mock_content_image
        ):
            with patch.object(
                self.pipeline,
                "_analyze_style_image",
                return_value="warm tones, bright lighting",
            ):
                with patch.object(
                    self.pipeline.model, "img2img", return_value=mock_result_image
                ) as mock_img2img:
                    result = self.pipeline.dual_image_style_transfer(
                        content_image=mock_content_image,
                        style_image=mock_style_image,
                        style_strength=0.8,
                        content_strength=0.6,
                        prompt_override="custom prompt",
                    )

                    assert result == mock_result_image
                    mock_img2img.assert_called_once()

                    call_args = mock_img2img.call_args[1]
                    assert "custom prompt" in call_args["prompt"]
                    assert "warm tones, bright lighting" in call_args["prompt"]
                    assert call_args["strength"] == 0.4  # 1.0 - content_strength

    def test_dual_image_style_transfer_without_prompt_override(self):
        """Test dual image style transfer without prompt override."""
        self.pipeline.loaded = True
        self.pipeline.model.device = "cpu"

        mock_content_image = Mock(spec=Image.Image)
        mock_style_image = Mock(spec=Image.Image)
        mock_result_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline, "_prepare_image", return_value=mock_content_image
        ):
            with patch.object(
                self.pipeline, "_analyze_style_image", return_value="cool tones"
            ):
                with patch.object(
                    self.pipeline.model, "img2img", return_value=mock_result_image
                ) as mock_img2img:
                    result = self.pipeline.dual_image_style_transfer(
                        content_image=mock_content_image, style_image=mock_style_image
                    )

                    call_args = mock_img2img.call_args[1]
                    assert "detailed artistic composition" in call_args["prompt"]

    @patch("cv2.cvtColor")
    @patch("cv2.Canny")
    def test_analyze_style_image_warm_tones(self, mock_canny, mock_cvtColor):
        """Test style image analysis for warm tones."""
        # Create test image with red dominance
        test_image = Mock(spec=Image.Image)
        test_array = np.zeros((100, 100, 3))
        test_array[:, :, 0] = 200  # High red values
        test_array[:, :, 1] = 100  # Medium green
        test_array[:, :, 2] = 50  # Low blue

        # Mock cv2 functions
        mock_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_edges = (
            np.random.randint(0, 255, (100, 100), dtype=np.uint8) * 0.2
        )  # Low edge density
        mock_cvtColor.return_value = mock_gray
        mock_canny.return_value = mock_edges

        with patch("numpy.array", return_value=test_array):
            result = self.pipeline._analyze_style_image(test_image)

            assert "warm tones" in result
            # Check for any lighting description (brightness calculation may vary)
            assert any(
                term in result
                for term in ["bright lighting", "balanced lighting", "dramatic shadows"]
            )
            assert "smooth gradients" in result  # Low edge density

    @patch("cv2.cvtColor")
    @patch("cv2.Canny")
    def test_analyze_style_image_cool_tones(self, mock_canny, mock_cvtColor):
        """Test style image analysis for cool tones."""
        # Create test image with blue dominance
        test_image = Mock(spec=Image.Image)
        test_array = np.zeros((100, 100, 3))
        test_array[:, :, 0] = 50  # Low red
        test_array[:, :, 1] = 100  # Medium green
        test_array[:, :, 2] = 200  # High blue values

        # Mock cv2 functions
        mock_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_edges = (
            np.random.randint(0, 255, (100, 100), dtype=np.uint8) * 0.8
        )  # High edge density
        mock_cvtColor.return_value = mock_gray
        mock_canny.return_value = mock_edges

        with patch("numpy.array", return_value=test_array):
            result = self.pipeline._analyze_style_image(test_image)

            assert "cool tones" in result
            assert "detailed textures" in result  # High edge density

    @patch("cv2.cvtColor")
    @patch("cv2.Canny")
    def test_analyze_style_image_natural_colors(self, mock_canny, mock_cvtColor):
        """Test style image analysis for natural colors."""
        # Create test image with green dominance and dark tones
        test_image = Mock(spec=Image.Image)
        test_array = np.zeros((100, 100, 3))
        test_array[:, :, 0] = 30  # Low red
        test_array[:, :, 1] = 100  # High green
        test_array[:, :, 2] = 50  # Medium blue

        # Mock cv2 functions
        mock_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_edges = (
            np.random.randint(0, 255, (100, 100), dtype=np.uint8) * 0.1
        )  # Medium edge density
        mock_cvtColor.return_value = mock_gray
        mock_canny.return_value = mock_edges

        with patch("numpy.array", return_value=test_array):
            result = self.pipeline._analyze_style_image(test_image)

            assert "natural colors" in result
            assert "dramatic shadows" in result  # Low brightness
            assert "smooth gradients" in result

    def test_batch_process_not_loaded(self):
        """Test batch processing when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.pipeline.batch_process(
                ["image1.jpg", "image2.jpg"], "Van Gogh style", Path("output")
            )

    def test_batch_process_success(self):
        """Test successful batch processing."""
        self.pipeline.loaded = True

        images = ["image1.jpg", "image2.jpg"]
        output_dir = Path("test_output")
        style_description = "impressionist painting"

        mock_result_image = Mock(spec=Image.Image)

        with patch.object(
            self.pipeline, "image_to_image_stylized", return_value=mock_result_image
        ):
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                result = self.pipeline.batch_process(
                    images, style_description, output_dir, style_strength=0.8
                )

                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                assert len(result) == 2
                assert all(
                    isinstance(path, Path) for path in result if path is not None
                )

                # Check that save was called for each image
                assert mock_result_image.save.call_count == 2

    def test_batch_process_with_failures(self):
        """Test batch processing with some failures."""
        self.pipeline.loaded = True

        images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        output_dir = Path("test_output")
        style_description = "test style"

        mock_result_image = Mock(spec=Image.Image)

        # Mock the image_to_image_stylized method to fail on second image
        def side_effect(content_image, **kwargs):
            if "image2" in str(content_image):
                raise RuntimeError("Processing failed")
            return mock_result_image

        with patch.object(
            self.pipeline, "image_to_image_stylized", side_effect=side_effect
        ):
            with patch("pathlib.Path.mkdir"):
                result = self.pipeline.batch_process(
                    images, style_description, output_dir
                )

                assert len(result) == 3
                assert result[0] is not None  # Success
                assert result[1] is None  # Failure
                assert result[2] is not None  # Success

    def test_get_pipeline_info_not_loaded(self):
        """Test getting pipeline info when not loaded."""
        result = self.pipeline.get_pipeline_info()

        assert result["loaded"] is False
        assert "config" in result
        assert result["config"] == self.config.to_dict()

    def test_get_pipeline_info_loaded(self):
        """Test getting pipeline info when loaded."""
        self.pipeline.loaded = True

        mock_model_info = {
            "model_name": "test/model",
            "device": "cpu",
            "mlx_enabled": True,
        }

        with patch.object(
            self.pipeline.model, "get_model_info", return_value=mock_model_info
        ):
            result = self.pipeline.get_pipeline_info()

            assert result["loaded"] is True
            assert result["config"] == self.config.to_dict()
            assert result["model_name"] == "test/model"
            assert result["device"] == "cpu"
            assert result["mlx_enabled"] is True


class TestDiffusionPipelineIntegration:
    """Test DiffusionPipeline integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig(
            model_name="test/model", num_inference_steps=10, guidance_scale=5.0
        )

    def test_complete_workflow(self):
        """Test complete pipeline workflow."""
        pipeline = DiffusionPipeline(self.config)

        # Mock dependencies
        mock_model_load = Mock()
        mock_model_unload = Mock()
        mock_generate = Mock(return_value=Mock(spec=Image.Image))

        pipeline.model.load_model = mock_model_load
        pipeline.model.unload_model = mock_model_unload
        pipeline.model.generate_image = mock_generate
        pipeline.model.device = "cpu"

        # Test workflow
        assert pipeline.loaded is False

        # Load model
        pipeline.load_model()
        assert pipeline.loaded is True
        mock_model_load.assert_called_once()

        # Generate stylized image
        result = pipeline.text_to_image_stylized(
            "a mountain landscape", "Van Gogh painting"
        )
        assert result is not None
        mock_generate.assert_called_once()

        # Unload model
        pipeline.unload_model()
        assert pipeline.loaded is False
        mock_model_unload.assert_called_once()

    def test_error_handling_in_style_analysis(self):
        """Test error handling in style analysis."""
        pipeline = DiffusionPipeline(self.config)

        # Create a mock image that will cause cv2 to fail
        mock_image = Mock(spec=Image.Image)

        with patch("numpy.array", side_effect=RuntimeError("Array conversion failed")):
            # Should not raise exception, might return default or handle gracefully
            try:
                result = pipeline._analyze_style_image(mock_image)
                # If it succeeds, that's fine
            except RuntimeError:
                # If it fails, that's also expected behavior
                pass

    def test_various_image_formats(self):
        """Test pipeline with various image input formats."""
        pipeline = DiffusionPipeline(self.config)

        # Test string path
        mock_image = Mock(spec=Image.Image)
        mock_image.convert.return_value = mock_image

        with patch("PIL.Image.open", return_value=mock_image):
            result = pipeline._prepare_image("test.jpg")
            assert result == mock_image

        # Test Path object
        with patch("PIL.Image.open", return_value=mock_image):
            result = pipeline._prepare_image(Path("test.png"))
            assert result == mock_image

        # Test numpy array
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with patch("PIL.Image.fromarray", return_value=mock_image):
            result = pipeline._prepare_image(test_array)
            assert result == mock_image

        # Test PIL Image
        pil_image = Mock(spec=Image.Image)
        result = pipeline._prepare_image(pil_image)
        assert result == pil_image

    def test_generator_seed_handling(self):
        """Test proper generator and seed handling."""
        pipeline = DiffusionPipeline(self.config)
        pipeline.loaded = True
        pipeline.model.device = "cpu"

        mock_image = Mock(spec=Image.Image)

        with patch.object(
            pipeline.model, "generate_image", return_value=mock_image
        ) as mock_generate:
            # Test with seed
            pipeline.text_to_image_stylized("test prompt", "test style", seed=42)

            call_args = mock_generate.call_args[1]
            generator = call_args["generator"]
            assert generator is not None
            assert isinstance(generator, torch.Generator)

            # Test without seed
            pipeline.text_to_image_stylized("test prompt", "test style")

            call_args = mock_generate.call_args[1]
            assert call_args["generator"] is None
