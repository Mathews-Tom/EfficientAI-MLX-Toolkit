"""Test style transfer functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from src.style_transfer.config import StyleTransferConfig
from src.style_transfer.pipeline import StyleTransferPipeline
from src.style_transfer.engine import StyleTransferEngine, StyleTransferResult


class TestStyleTransferPipeline:
    """Test StyleTransferPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StyleTransferConfig()
        self.pipeline = StyleTransferPipeline(self.config)
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        assert self.pipeline.config == self.config
        assert self.pipeline.diffusion_pipeline is None
        assert self.pipeline._device in ["cpu", "mps", "cuda"]
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        config = StyleTransferConfig(device="auto")
        pipeline = StyleTransferPipeline(config)
        
        device = pipeline._get_device()
        assert device in ["cpu", "mps", "cuda"]
    
    def test_get_device_manual(self):
        """Test manual device selection."""
        config = StyleTransferConfig(device="cpu")
        pipeline = StyleTransferPipeline(config)
        
        device = pipeline._get_device()
        assert device == "cpu"
    
    @patch('src.style_transfer.pipeline.Image.open')
    def test_prepare_image_from_path(self, mock_open):
        """Test image preparation from file path."""
        # Mock PIL Image with proper attributes
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        mock_image.size = (256, 256)
        mock_image.width = 256
        mock_image.height = 256
        mock_image.resize.return_value = mock_image
        mock_image.thumbnail.return_value = None  # thumbnail modifies in place
        mock_open.return_value = mock_image
        
        # Configure the pipeline to not preserve aspect ratio for simpler testing
        self.pipeline.config.preserve_aspect_ratio = False
        
        result = self.pipeline._prepare_image("test.jpg")
        
        mock_open.assert_called_once_with("test.jpg")
        mock_image.convert.assert_called_once_with("RGB")
    
    def test_prepare_image_from_array(self):
        """Test image preparation from numpy array."""
        # Create test array
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with patch('PIL.Image.fromarray') as mock_fromarray:
            mock_image = Mock()
            mock_image.size = (100, 100)
            mock_image.width = 100
            mock_image.height = 100
            mock_image.resize.return_value = mock_image
            mock_image.thumbnail.return_value = None
            mock_fromarray.return_value = mock_image
            
            # Configure the pipeline to not preserve aspect ratio for simpler testing
            self.pipeline.config.preserve_aspect_ratio = False
            
            result = self.pipeline._prepare_image(test_array)
            
            mock_fromarray.assert_called_once()
    
    def test_postprocess_image_upscale(self):
        """Test image post-processing with upscaling."""
        config = StyleTransferConfig(upscale_factor=2.0)
        pipeline = StyleTransferPipeline(config)
        
        mock_image = Mock()
        mock_image.width = 100
        mock_image.height = 100
        mock_image.resize.return_value = mock_image
        
        result = pipeline._postprocess_image(mock_image)
        
        mock_image.resize.assert_called_once_with((200, 200), Image.Resampling.LANCZOS)
    
    def test_postprocess_image_no_upscale(self):
        """Test image post-processing without upscaling."""
        config = StyleTransferConfig(upscale_factor=1.0)
        pipeline = StyleTransferPipeline(config)
        
        mock_image = Mock()
        result = pipeline._postprocess_image(mock_image)
        
        # Should return original image without modification
        assert result == mock_image
    
    def test_transfer_style_no_inputs(self):
        """Test style transfer with no style inputs."""
        mock_image = Mock()
        
        with pytest.raises(ValueError, match="Either style_image or style_description must be provided"):
            self.pipeline.transfer_style(mock_image)
    
    def test_transfer_style_unknown_method(self):
        """Test style transfer with unknown method."""
        config = StyleTransferConfig(method="unknown", preserve_aspect_ratio=False)
        pipeline = StyleTransferPipeline(config)
        
        # Create a proper mock image with required attributes
        mock_image = Mock()
        mock_image.size = (256, 256)
        mock_image.width = 256
        mock_image.height = 256
        mock_image.resize.return_value = mock_image
        
        with pytest.raises(ValueError, match="Unknown method"):
            pipeline.transfer_style(mock_image, style_description="test")
    
    @patch('src.style_transfer.pipeline.StyleTransferPipeline._prepare_image')
    def test_neural_style_transfer_not_implemented(self, mock_prepare):
        """Test neural style transfer raises not implemented."""
        config = StyleTransferConfig(method="neural_style")
        pipeline = StyleTransferPipeline(config)
        
        # Mock the prepare_image method to return a simple mock
        mock_image = Mock()
        mock_prepare.return_value = mock_image
        
        with pytest.raises(NotImplementedError, match="Neural style transfer method not yet implemented"):
            pipeline.transfer_style("fake_image", style_image="fake_style")


class TestStyleTransferEngine:
    """Test StyleTransferEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StyleTransferConfig()
        self.engine = StyleTransferEngine(self.config)
    
    def test_engine_creation(self):
        """Test engine creation."""
        assert self.engine.config == self.config
        assert isinstance(self.engine.pipeline, StyleTransferPipeline)
        assert "total_processed" in self.engine.processing_stats
    
    @patch('psutil.Process')
    def test_get_memory_usage_cpu(self, mock_process_class):
        """Test memory usage calculation on CPU."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process_class.return_value = mock_process
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                memory = self.engine._get_memory_usage()
        
        assert memory == 100.0  # 100MB
    
    def test_reset_stats(self):
        """Test statistics reset."""
        # Modify stats
        self.engine.processing_stats["total_processed"] = 5
        self.engine.processing_stats["total_time"] = 10.0
        
        # Reset
        self.engine.reset_stats()
        
        # Check reset
        assert self.engine.processing_stats["total_processed"] == 0
        assert self.engine.processing_stats["total_time"] == 0.0
        assert self.engine.processing_stats["average_time"] == 0.0
        assert self.engine.processing_stats["peak_memory"] == 0.0
    
    def test_get_engine_stats(self):
        """Test engine statistics retrieval."""
        stats = self.engine.get_engine_stats()
        
        assert "processing_stats" in stats
        assert "pipeline_info" in stats
        assert "config" in stats
        assert stats["config"] == self.config.to_dict()


class TestStyleTransferResult:
    """Test StyleTransferResult dataclass."""
    
    def test_result_creation(self):
        """Test result creation."""
        mock_image = Mock()
        config = StyleTransferConfig()
        metadata = {"test": "value"}
        
        result = StyleTransferResult(
            image=mock_image,
            processing_time=1.5,
            memory_used=100.0,
            config_used=config,
            metadata=metadata
        )
        
        assert result.image == mock_image
        assert result.processing_time == 1.5
        assert result.memory_used == 100.0
        assert result.config_used == config
        assert result.metadata == metadata


class TestStyleAnalysis:
    """Test style analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StyleTransferConfig()
        self.pipeline = StyleTransferPipeline(self.config)
    
    def test_analyze_style_image(self):
        """Test style image analysis."""
        # Create test image array
        test_image = Image.new('RGB', (100, 100), color='red')
        
        description = self.pipeline._analyze_style_image(test_image)
        
        assert isinstance(description, str)
        assert len(description) > 0
        # Should contain some descriptive terms
        assert any(term in description for term in ['warm', 'cool', 'bright', 'dark'])


class TestImageProcessing:
    """Test image processing utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StyleTransferConfig(
            output_resolution=(256, 256),
            preserve_aspect_ratio=True
        )
        self.pipeline = StyleTransferPipeline(self.config)
    
    def test_prepare_image_aspect_ratio_preserved(self):
        """Test image preparation with aspect ratio preservation."""
        # Create mock image with different aspect ratio
        mock_image = Mock()
        mock_image.width = 400
        mock_image.height = 200
        mock_image.size = (400, 200)
        mock_image.convert.return_value = mock_image
        mock_image.thumbnail.return_value = None
        
        # Mock the new image creation
        with patch('PIL.Image.new') as mock_new:
            with patch('PIL.Image.open', return_value=mock_image):
                mock_new_image = Mock()
                mock_new_image.paste.return_value = None
                mock_new.return_value = mock_new_image
                
                result = self.pipeline._prepare_image("test.jpg")
                
                # Should create new image with target size
                mock_new.assert_called_once_with("RGB", (256, 256), (255, 255, 255))
                mock_new_image.paste.assert_called_once()
    
    def test_prepare_image_no_aspect_ratio_preservation(self):
        """Test image preparation without aspect ratio preservation."""
        config = StyleTransferConfig(
            output_resolution=(256, 256),
            preserve_aspect_ratio=False
        )
        pipeline = StyleTransferPipeline(config)
        
        mock_image = Mock()
        mock_image.size = (400, 300)
        mock_image.width = 400
        mock_image.height = 300
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image
        
        with patch('PIL.Image.open', return_value=mock_image):
            result = pipeline._prepare_image("test.jpg")
            
            mock_image.resize.assert_called_once_with((256, 256), Image.Resampling.LANCZOS)