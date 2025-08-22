"""Style transfer engine for advanced operations."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import StyleTransferConfig
from .pipeline import StyleTransferPipeline


@dataclass
class StyleTransferResult:
    """Result of a style transfer operation."""

    image: Image.Image
    processing_time: float
    memory_used: float
    config_used: StyleTransferConfig
    metadata: dict[str, any]


class StyleTransferEngine:
    """Advanced style transfer engine with optimization and batch processing."""

    def __init__(self, config: StyleTransferConfig):
        self.config = config
        self.pipeline = StyleTransferPipeline(config)
        self.processing_stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "peak_memory": 0.0,
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        elif torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1024**2
        else:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024**2

    def transfer_style_advanced(
        self,
        content_image: Image.Image | np.ndarray | str | Path,
        style_image: Image.Image | np.ndarray | str | Path | None = None,
        style_description: str | None = None,
        seed: int | None = None,
        track_metrics: bool = True,
    ) -> StyleTransferResult:
        """Perform style transfer with detailed metrics tracking."""
        start_time = time.time()
        start_memory = self._get_memory_usage() if track_metrics else 0.0

        try:
            # Perform style transfer
            result_image = self.pipeline.transfer_style(
                content_image=content_image,
                style_image=style_image,
                style_description=style_description,
                seed=seed,
            )

            processing_time = time.time() - start_time
            peak_memory = self._get_memory_usage() if track_metrics else 0.0
            memory_used = peak_memory - start_memory

            # Update stats
            if track_metrics:
                self.processing_stats["total_processed"] += 1
                self.processing_stats["total_time"] += processing_time
                self.processing_stats["average_time"] = (
                    self.processing_stats["total_time"]
                    / self.processing_stats["total_processed"]
                )
                self.processing_stats["peak_memory"] = max(
                    self.processing_stats["peak_memory"], peak_memory
                )

            # Create metadata
            metadata = {
                "seed": seed,
                "has_style_image": style_image is not None,
                "has_style_description": style_description is not None,
                "output_size": result_image.size,
                "method": self.config.method,
            }

            return StyleTransferResult(
                image=result_image,
                processing_time=processing_time,
                memory_used=memory_used,
                config_used=self.config,
                metadata=metadata,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(
                f"Style transfer failed after {processing_time:.2f}s: {e}"
            ) from e

    def batch_transfer_parallel(
        self,
        content_images: list[Image.Image | np.ndarray | str | Path],
        style_image: Image.Image | np.ndarray | str | Path | None = None,
        style_description: str | None = None,
        max_workers: int = 2,
        output_dir: Path | None = None,
        save_images: bool = True,
    ) -> list[StyleTransferResult]:
        """Process multiple images in parallel."""
        if output_dir and save_images:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = [None] * len(content_images)

        def process_single_image(idx_and_image):
            idx, image_path = idx_and_image
            try:
                # Create a separate pipeline for each worker to avoid conflicts
                worker_pipeline = StyleTransferPipeline(self.config)

                result_image = worker_pipeline.transfer_style(
                    content_image=image_path,
                    style_image=style_image,
                    style_description=style_description,
                )

                # Save if requested
                if output_dir and save_images:
                    output_path = (
                        output_dir
                        / f"styled_{idx:03d}.{self.config.output_format.lower()}"
                    )
                    result_image.save(
                        output_path,
                        format=self.config.output_format,
                        quality=self.config.quality,
                    )

                # Clean up worker pipeline
                worker_pipeline.cleanup()

                return idx, StyleTransferResult(
                    image=result_image,
                    processing_time=0.0,  # Not tracked in parallel mode
                    memory_used=0.0,  # Not tracked in parallel mode
                    config_used=self.config,
                    metadata={"batch_index": idx, "parallel_processing": True},
                )

            except Exception as e:
                print(f"Failed to process image {idx}: {e}")
                return idx, None

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, (i, img)): i
                for i, img in enumerate(content_images)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

                if result is not None:
                    print(f"Completed image {idx + 1}/{len(content_images)}")
                else:
                    print(f"Failed image {idx + 1}/{len(content_images)}")

        return results

    def create_style_grid(
        self,
        content_image: Image.Image | np.ndarray | str | Path,
        style_images: list[Image.Image | np.ndarray | str | Path],
        grid_size: tuple[int, int] | None = None,
        output_path: Path | None = None,
    ) -> Image.Image:
        """Create a grid showing one content image with multiple styles."""
        if grid_size is None:
            # Auto-calculate grid size
            num_styles = len(style_images)
            cols = int(np.ceil(np.sqrt(num_styles + 1)))  # +1 for original
            rows = int(np.ceil((num_styles + 1) / cols))
            grid_size = (rows, cols)

        # Prepare content image
        content_img = self.pipeline._prepare_image(content_image)
        cell_size = content_img.size

        # Create grid image
        grid_width = grid_size[1] * cell_size[0]
        grid_height = grid_size[0] * cell_size[1]
        grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        # Place original content image in top-left
        grid_image.paste(content_img, (0, 0))

        # Process and place styled versions
        for i, style_img in enumerate(style_images[: grid_size[0] * grid_size[1] - 1]):
            row = (i + 1) // grid_size[1]
            col = (i + 1) % grid_size[1]

            x_pos = col * cell_size[0]
            y_pos = row * cell_size[1]

            try:
                styled_img = self.pipeline.transfer_style(
                    content_image=content_image, style_image=style_img
                )
                grid_image.paste(styled_img, (x_pos, y_pos))

            except Exception as e:
                print(f"Failed to process style {i}: {e}")
                # Place a placeholder or skip
                continue

        # Save if path provided
        if output_path:
            grid_image.save(
                output_path,
                format=self.config.output_format,
                quality=self.config.quality,
            )

        return grid_image

    def benchmark_performance(
        self,
        test_images: list[Image.Image | np.ndarray | str | Path],
        style_image: Image.Image | np.ndarray | str | Path,
        iterations: int = 3,
        warmup_iterations: int = 1,
    ) -> dict[str, any]:
        """Benchmark style transfer performance."""
        print(f"Benchmarking with {len(test_images)} images, {iterations} iterations")

        # Warmup runs
        print("Performing warmup runs...")
        for _ in range(warmup_iterations):
            for img in test_images[
                : min(2, len(test_images))
            ]:  # Use max 2 images for warmup
                try:
                    self.pipeline.transfer_style(
                        content_image=img, style_image=style_image
                    )
                except Exception as e:
                    print(f"Warmup failed: {e}")

        # Clear cache before benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Actual benchmark
        times = []
        memory_usage = []

        for iteration in range(iterations):
            print(f"Benchmark iteration {iteration + 1}/{iterations}")
            iteration_times = []
            iteration_memory = []

            for img in test_images:
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    self.pipeline.transfer_style(
                        content_image=img, style_image=style_image
                    )

                    end_time = time.time()
                    peak_memory = self._get_memory_usage()

                    iteration_times.append(end_time - start_time)
                    iteration_memory.append(peak_memory - start_memory)

                except Exception as e:
                    print(f"Benchmark failed on image: {e}")
                    continue

            times.extend(iteration_times)
            memory_usage.extend(iteration_memory)

        if not times:
            raise RuntimeError("All benchmark runs failed")

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        throughput = len(test_images) * iterations / sum(times)

        avg_memory = np.mean(memory_usage) if memory_usage else 0
        peak_memory = np.max(memory_usage) if memory_usage else 0

        return {
            "total_images": len(test_images) * iterations,
            "total_time": sum(times),
            "avg_time_per_image": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput_images_per_sec": throughput,
            "avg_memory_usage_mb": avg_memory,
            "peak_memory_usage_mb": peak_memory,
            "config": self.config.to_dict(),
            "device": self.pipeline._device,
        }

    def get_engine_stats(self) -> dict[str, any]:
        """Get engine processing statistics."""
        return {
            "processing_stats": self.processing_stats.copy(),
            "pipeline_info": self.pipeline.get_pipeline_info(),
            "config": self.config.to_dict(),
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "peak_memory": 0.0,
        }

    def cleanup(self) -> None:
        """Clean up engine resources."""
        self.pipeline.cleanup()
