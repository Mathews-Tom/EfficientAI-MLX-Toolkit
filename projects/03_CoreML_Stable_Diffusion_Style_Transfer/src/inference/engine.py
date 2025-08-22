"""Inference engine for style transfer models."""

import time
from pathlib import Path
from dataclasses import dataclass

from PIL import Image
import numpy as np

from .config import InferenceConfig


@dataclass
class InferenceResult:
    """Result from inference operation."""
    
    image: Image.Image
    processing_time: float
    metadata: dict[str, any]


class InferenceEngine:
    """Engine for running inference on style transfer models."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.config.validate()
        self.model = None
        self.loaded = False
        
    def load_model(self) -> None:
        """Load the model for inference."""
        if self.config.model_path is None:
            raise ValueError("model_path must be specified")
            
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"Loading model from: {model_path}")
        
        # Determine model type and load accordingly
        if str(model_path).endswith('.mlpackage') or str(model_path).endswith('.mlmodel'):
            self._load_coreml_model(model_path)
        else:
            self._load_pytorch_model(model_path)
            
        self.loaded = True
        print("Model loaded successfully")
    
    def _load_coreml_model(self, model_path: Path) -> None:
        """Load Core ML model."""
        try:
            import coremltools as ct
            self.model = ct.models.MLModel(str(model_path))
            self.model_type = "coreml"
        except ImportError:
            raise ImportError("coremltools is required for Core ML models")
    
    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model."""
        import torch
        self.model = torch.load(model_path, map_location='cpu')
        self.model_type = "pytorch"
    
    def predict(
        self, 
        content_image: Image.Image | np.ndarray,
        style_image: Image.Image | np.ndarray | None = None,
        style_description: str | None = None
    ) -> InferenceResult:
        """Run inference on input data."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Placeholder for actual inference
        # This would integrate with the style transfer pipeline
        result_image = self._run_inference(content_image, style_image, style_description)
        
        processing_time = time.time() - start_time
        
        metadata = {
            "processing_time": processing_time,
            "model_type": self.model_type,
            "has_style_image": style_image is not None,
            "has_style_description": style_description is not None
        }
        
        return InferenceResult(
            image=result_image,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _run_inference(
        self,
        content_image: Image.Image | np.ndarray,
        style_image: Image.Image | np.ndarray | None,
        style_description: str | None
    ) -> Image.Image:
        """Run the actual inference."""
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Preprocess inputs
        # 2. Run model inference
        # 3. Postprocess outputs
        
        if isinstance(content_image, np.ndarray):
            content_image = Image.fromarray(content_image)
            
        # For now, return the content image as a placeholder
        return content_image
    
    def benchmark(
        self,
        test_images_dir: Path,
        iterations: int = 10,
        output_dir: Path | None = None
    ) -> dict[str, any]:
        """Benchmark inference performance."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        test_images_dir = Path(test_images_dir)
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
        # Get test images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(test_images_dir.glob(ext))
        
        if not image_files:
            raise ValueError("No test images found in directory")
        
        print(f"Benchmarking with {len(image_files)} images, {iterations} iterations")
        
        times = []
        memory_usage = []
        
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")
            
            for img_path in image_files:
                try:
                    # Load image
                    content_image = Image.open(img_path).convert('RGB')
                    
                    # Run inference
                    start_time = time.time()
                    result = self.predict(content_image, style_description="benchmark test")
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    # Save result if output directory specified
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f"result_{iteration}_{img_path.stem}.png"
                        result.image.save(output_path)
                        
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
                    continue
        
        if not times:
            raise RuntimeError("No successful benchmark runs")
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        throughput = len(image_files) * iterations / sum(times)
        
        return {
            "total_images": len(image_files) * iterations,
            "total_time": sum(times),
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": throughput,
            "config": self.config.to_dict()
        }
    
    def get_model_info(self) -> dict[str, any]:
        """Get information about the loaded model."""
        if not self.loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_type": self.model_type,
            "model_path": str(self.config.model_path),
            "config": self.config.to_dict()
        }
        
        return info
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.loaded = False
            print("Model unloaded")