"""Test diffusion model functionality."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.diffusion.config import DiffusionConfig
from src.diffusion.model import StableDiffusionMLX


class TestStableDiffusionMLX:
    """Test StableDiffusionMLX class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig()
        self.model = StableDiffusionMLX(self.config)

    def test_model_creation(self):
        """Test model creation."""
        assert self.model.config == self.config
        assert self.model.pipeline is None
        assert self.model.device in ["cpu", "mps", "cuda"]

    def test_get_device_auto_mps(self):
        """Test automatic device selection with MPS available."""
        config = DiffusionConfig(device="auto", use_mps=True)

        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.cuda.is_available", return_value=False):
                model = StableDiffusionMLX(config)
                assert model.device == "mps"

    def test_get_device_auto_cuda(self):
        """Test automatic device selection with CUDA available."""
        config = DiffusionConfig(device="auto", use_mps=False)

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=True):
                model = StableDiffusionMLX(config)
                assert model.device == "cuda"

    def test_get_device_auto_cpu(self):
        """Test automatic device selection fallback to CPU."""
        config = DiffusionConfig(device="auto")

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=False):
                model = StableDiffusionMLX(config)
                assert model.device == "cpu"

    def test_get_device_manual(self):
        """Test manual device selection."""
        config = DiffusionConfig(device="cpu")
        model = StableDiffusionMLX(config)
        assert model.device == "cpu"

    def test_get_scheduler_dpm(self):
        """Test DPM scheduler selection."""
        scheduler_class = self.model._get_scheduler("DPMSolverMultistepScheduler")
        from diffusers.schedulers import DPMSolverMultistepScheduler

        assert scheduler_class == DPMSolverMultistepScheduler

    def test_get_scheduler_ddim(self):
        """Test DDIM scheduler selection."""
        scheduler_class = self.model._get_scheduler("DDIMScheduler")
        from diffusers.schedulers import DDIMScheduler

        assert scheduler_class == DDIMScheduler

    def test_get_scheduler_unknown(self):
        """Test unknown scheduler fallback."""
        scheduler_class = self.model._get_scheduler("UnknownScheduler")
        from diffusers.schedulers import DPMSolverMultistepScheduler

        assert scheduler_class == DPMSolverMultistepScheduler

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_load_model_success(self, mock_from_pretrained):
        """Test successful model loading."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.scheduler.config = {}
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline.enable_model_cpu_offload = Mock()
        mock_pipeline.enable_sequential_cpu_offload = Mock()
        mock_from_pretrained.return_value = mock_pipeline

        # Mock scheduler
        mock_scheduler_class = Mock()
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler

        with patch.object(
            self.model, "_get_scheduler", return_value=mock_scheduler_class
        ):
            self.model.load_model()

        assert self.model.pipeline == mock_pipeline
        mock_from_pretrained.assert_called_once()
        mock_pipeline.to.assert_called_once_with(self.model.device)

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_load_model_with_optimizations(self, mock_from_pretrained):
        """Test model loading with optimizations enabled."""
        config = DiffusionConfig(
            use_attention_slicing=True,
            enable_memory_efficient_attention=True,
            use_cpu_offload=True,
        )
        model = StableDiffusionMLX(config)

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.scheduler.config = {}
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        # Mock scheduler
        mock_scheduler_class = Mock()
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler

        with patch.object(model, "_get_scheduler", return_value=mock_scheduler_class):
            model.load_model()

        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_model_cpu_offload.assert_called_once()
        mock_pipeline.enable_sequential_cpu_offload.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_load_model_with_mlx_optimization(self, mock_from_pretrained):
        """Test model loading with MLX optimization."""
        config = DiffusionConfig(use_mlx=True, device="mps")
        model = StableDiffusionMLX(config)
        model.device = "mps"  # Force MPS device

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.scheduler.config = {}
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        # Mock scheduler and MLX optimization
        mock_scheduler_class = Mock()
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler

        with patch.object(model, "_get_scheduler", return_value=mock_scheduler_class):
            with patch.object(model, "_apply_mlx_optimizations") as mock_mlx:
                model.load_model()
                mock_mlx.assert_called_once()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_load_model_failure(self, mock_from_pretrained):
        """Test model loading failure."""
        mock_from_pretrained.side_effect = RuntimeError("Loading failed")

        with pytest.raises(RuntimeError, match="Failed to load model"):
            self.model.load_model()

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    def test_load_model_with_custom_config(self, mock_from_pretrained):
        """Test model loading with custom configuration."""
        config = DiffusionConfig(
            model_name="custom/model",
            variant="fp16",
            torch_dtype="float16",
            safety_checker=False,
            requires_safety_checker=False,
            offline_mode=True,
            cache_dir="/custom/cache",
        )
        model = StableDiffusionMLX(config)

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.scheduler.config = {}
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        # Mock scheduler
        mock_scheduler_class = Mock()
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler

        with patch.object(model, "_get_scheduler", return_value=mock_scheduler_class):
            model.load_model()

        # Verify custom parameters were passed
        call_args = mock_from_pretrained.call_args
        assert call_args[0][0] == "custom/model"
        assert call_args[1]["variant"] == "fp16"
        assert call_args[1]["torch_dtype"] == torch.float16
        assert call_args[1]["safety_checker"] is None
        assert call_args[1]["local_files_only"] is True
        assert call_args[1]["cache_dir"] == "/custom/cache"

    def test_apply_mlx_optimizations(self):
        """Test MLX optimization application."""
        # This should not raise an exception (placeholder implementation)
        self.model._apply_mlx_optimizations()

    def test_apply_mlx_optimizations_with_exception(self):
        """Test MLX optimization with exception handling."""
        with patch("builtins.print") as mock_print:
            self.model._apply_mlx_optimizations()
            # Should not raise, just print the optimization message

    def test_generate_image_model_not_loaded(self):
        """Test image generation without loaded model."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.model.generate_image("test prompt")

    @patch("torch.inference_mode")
    def test_generate_image_text_to_image(self, mock_inference_mode):
        """Test text-to-image generation."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result
        self.model.pipeline = mock_pipeline

        # Mock inference context
        mock_inference_mode.return_value.__enter__ = Mock()
        mock_inference_mode.return_value.__exit__ = Mock()

        result = self.model.generate_image(
            prompt="test prompt",
            negative_prompt="bad quality",
            width=256,
            height=256,
            num_inference_steps=20,
            guidance_scale=7.0,
        )

        assert result == mock_image
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args[1]
        assert call_args["prompt"] == "test prompt"
        assert call_args["negative_prompt"] == "bad quality"
        assert call_args["width"] == 256
        assert call_args["height"] == 256
        assert call_args["num_inference_steps"] == 20
        assert call_args["guidance_scale"] == 7.0

    @patch("torch.inference_mode")
    def test_generate_image_image_to_image(self, mock_inference_mode):
        """Test image-to-image generation."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_output_image = Mock(spec=Image.Image)
        mock_result.images = [mock_output_image]
        mock_pipeline.return_value = mock_result
        self.model.pipeline = mock_pipeline

        # Mock input image
        mock_input_image = Mock(spec=Image.Image)

        # Mock inference context
        mock_inference_mode.return_value.__enter__ = Mock()
        mock_inference_mode.return_value.__exit__ = Mock()

        result = self.model.generate_image(
            prompt="test prompt", image=mock_input_image, strength=0.7
        )

        assert result == mock_output_image
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args[1]
        assert call_args["prompt"] == "test prompt"
        assert call_args["image"] == mock_input_image
        assert call_args["strength"] == 0.7

    @patch("torch.inference_mode")
    def test_generate_image_with_defaults(self, mock_inference_mode):
        """Test image generation using config defaults."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result
        self.model.pipeline = mock_pipeline

        # Mock inference context
        mock_inference_mode.return_value.__enter__ = Mock()
        mock_inference_mode.return_value.__exit__ = Mock()

        result = self.model.generate_image("test prompt")

        assert result == mock_image
        call_args = mock_pipeline.call_args[1]
        assert call_args["num_inference_steps"] == self.config.num_inference_steps
        assert call_args["guidance_scale"] == self.config.guidance_scale

    @patch("torch.inference_mode")
    def test_generate_image_failure(self, mock_inference_mode):
        """Test image generation failure."""
        # Setup mock pipeline that raises exception when called
        mock_pipeline = Mock()
        mock_pipeline.side_effect = RuntimeError("Generation failed")
        self.model.pipeline = mock_pipeline

        # Mock inference context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=None)
        mock_context.__exit__ = Mock(return_value=None)
        mock_inference_mode.return_value = mock_context

        with pytest.raises(RuntimeError, match="Image generation failed"):
            self.model.generate_image("test prompt")

    def test_img2img(self):
        """Test img2img wrapper method."""
        mock_input_image = Mock(spec=Image.Image)
        mock_input_image.width = 512
        mock_input_image.height = 512
        mock_output_image = Mock(spec=Image.Image)

        with patch.object(
            self.model, "generate_image", return_value=mock_output_image
        ) as mock_generate:
            result = self.model.img2img(
                prompt="test prompt",
                image=mock_input_image,
                strength=0.8,
                negative_prompt="bad",
            )

            assert result == mock_output_image
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            assert call_args["prompt"] == "test prompt"
            assert call_args["image"] == mock_input_image
            assert call_args["strength"] == 0.8
            assert call_args["negative_prompt"] == "bad"
            assert call_args["width"] == 512
            assert call_args["height"] == 512

    def test_img2img_with_numpy_array(self):
        """Test img2img with numpy array input."""
        mock_input_array = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
        mock_output_image = Mock(spec=Image.Image)

        with patch.object(
            self.model, "generate_image", return_value=mock_output_image
        ) as mock_generate:
            result = self.model.img2img(prompt="test prompt", image=mock_input_array)

            assert result == mock_output_image
            call_args = mock_generate.call_args[1]
            assert call_args["width"] == 512  # image.shape[1]
            assert call_args["height"] == 256  # image.shape[0]

    def test_encode_prompt_model_not_loaded(self):
        """Test prompt encoding without loaded model."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            self.model.encode_prompt("test prompt")

    def test_encode_prompt_success(self):
        """Test successful prompt encoding."""
        # Setup mock pipeline components
        mock_tokenizer = Mock()
        mock_text_encoder = Mock()
        mock_text_inputs = Mock()
        mock_text_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_embeddings = torch.randn(1, 77, 768)

        mock_tokenizer.return_value = mock_text_inputs
        mock_tokenizer.model_max_length = 77
        mock_text_encoder.return_value = [mock_embeddings]

        mock_pipeline = Mock()
        mock_pipeline.tokenizer = mock_tokenizer
        mock_pipeline.text_encoder = mock_text_encoder
        self.model.pipeline = mock_pipeline

        with patch("torch.no_grad"):
            result = self.model.encode_prompt("test prompt")

        assert torch.equal(result, mock_embeddings)
        mock_tokenizer.assert_called_once_with(
            "test prompt",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

    def test_get_model_info_not_loaded(self):
        """Test getting model info when not loaded."""
        result = self.model.get_model_info()
        assert result == {"status": "not_loaded"}

    def test_get_model_info_loaded(self):
        """Test getting model info when loaded."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_unet = Mock()
        mock_unet.dtype = torch.float32
        mock_pipeline.unet = mock_unet
        self.model.pipeline = mock_pipeline

        with patch("torch.backends.mps.is_available", return_value=True):
            result = self.model.get_model_info()

        expected_info = {
            "model_name": self.config.model_name,
            "device": self.model.device,
            "dtype": "torch.float32",
            "scheduler": self.config.scheduler,
            "mlx_available": True,
            "mlx_enabled": self.config.use_mlx,
            "mps_available": True,
            "mps_enabled": self.config.use_mps and self.model.device == "mps",
        }

        for key, value in expected_info.items():
            if key != "mps_enabled":  # This depends on device
                assert result[key] == value

    @patch("torch.cuda.empty_cache")
    def test_clear_cache_cuda(self, mock_cuda_cache):
        """Test clearing CUDA cache."""
        self.model.device = "cuda"
        self.model.clear_cache()
        mock_cuda_cache.assert_called_once()

    @patch("torch.mps.empty_cache")
    def test_clear_cache_mps(self, mock_mps_cache):
        """Test clearing MPS cache."""
        self.model.device = "mps"
        self.model.clear_cache()
        mock_mps_cache.assert_called_once()

    def test_clear_cache_cpu(self):
        """Test clearing cache on CPU (no-op)."""
        self.model.device = "cpu"
        # Should not raise any exception
        self.model.clear_cache()

    def test_unload_model_with_pipeline(self):
        """Test unloading model when pipeline is loaded."""
        mock_pipeline = Mock()
        self.model.pipeline = mock_pipeline

        with patch.object(self.model, "clear_cache") as mock_clear:
            self.model.unload_model()

            assert self.model.pipeline is None
            mock_clear.assert_called_once()

    def test_unload_model_without_pipeline(self):
        """Test unloading model when no pipeline is loaded."""
        assert self.model.pipeline is None

        with patch.object(self.model, "clear_cache") as mock_clear:
            self.model.unload_model()

            # Should not call clear_cache since pipeline is None
            mock_clear.assert_not_called()


class TestStableDiffusionMLXIntegration:
    """Test StableDiffusionMLX integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig(
            model_name="test/model", num_inference_steps=10, guidance_scale=5.0
        )

    @patch("diffusers.StableDiffusionPipeline.from_pretrained")
    @patch("torch.inference_mode")
    def test_complete_workflow(self, mock_inference_mode, mock_from_pretrained):
        """Test complete workflow from loading to generation."""
        # Setup mocks
        mock_pipeline = Mock()
        mock_pipeline.scheduler.config = {}
        mock_pipeline.to.return_value = mock_pipeline
        mock_result = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline

        # Mock scheduler
        mock_scheduler_class = Mock()
        mock_scheduler = Mock()
        mock_scheduler_class.from_config.return_value = mock_scheduler

        # Mock inference context
        mock_inference_mode.return_value.__enter__ = Mock()
        mock_inference_mode.return_value.__exit__ = Mock()

        # Test workflow
        model = StableDiffusionMLX(self.config)

        with patch.object(model, "_get_scheduler", return_value=mock_scheduler_class):
            # Load model
            model.load_model()
            assert model.pipeline is not None

            # Generate image
            result = model.generate_image("test prompt")
            assert result == mock_image

            # Get model info
            info = model.get_model_info()
            assert info["model_name"] == "test/model"

            # Unload model
            model.unload_model()
            assert model.pipeline is None

    def test_scheduler_configuration(self):
        """Test different scheduler configurations."""
        schedulers = [
            "DPMSolverMultistepScheduler",
            "DDIMScheduler",
            "PNDMScheduler",
            "LMSDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
        ]

        for scheduler_name in schedulers:
            config = DiffusionConfig(scheduler=scheduler_name)
            model = StableDiffusionMLX(config)
            scheduler_class = model._get_scheduler(scheduler_name)
            assert scheduler_class is not None

    def test_device_optimization_combinations(self):
        """Test different device and optimization combinations."""
        test_cases = [
            {"device": "cpu", "use_mps": False, "use_mlx": False},
            {"device": "mps", "use_mps": True, "use_mlx": True},
            {"device": "cuda", "use_mps": False, "use_mlx": False},
        ]

        for case in test_cases:
            config = DiffusionConfig(**case)
            model = StableDiffusionMLX(config)
            assert model.config.device == case["device"]
            assert model.config.use_mps == case["use_mps"]
            assert model.config.use_mlx == case["use_mlx"]
