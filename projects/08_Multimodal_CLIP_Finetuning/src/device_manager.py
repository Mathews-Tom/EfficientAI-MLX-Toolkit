#!/usr/bin/env python3
"""
Device manager for MPS (Metal Performance Shaders) backend support.

Handles detection and configuration of Apple Silicon GPU acceleration
using PyTorch MPS backend.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and MPS backend configuration.

    Handles automatic detection of Apple Silicon hardware and MPS availability,
    with graceful fallback to CPU if MPS is unavailable.
    """

    def __init__(self, use_mps: bool = True) -> None:
        """Initialize device manager.

        Args:
            use_mps: Whether to attempt using MPS backend if available
        """
        self.use_mps = use_mps
        self._device: torch.device | None = None
        self._is_apple_silicon: bool | None = None
        self._is_mps_available: bool | None = None

    @property
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon hardware.

        Returns:
            True if running on Apple Silicon (M1/M2/M3), False otherwise
        """
        if self._is_apple_silicon is None:
            self._is_apple_silicon = self._detect_apple_silicon()
        return self._is_apple_silicon

    @property
    def is_mps_available(self) -> bool:
        """Check if MPS backend is available.

        Returns:
            True if MPS backend is available, False otherwise
        """
        if self._is_mps_available is None:
            self._is_mps_available = self._check_mps_availability()
        return self._is_mps_available

    @property
    def device(self) -> torch.device:
        """Get the selected device for computation.

        Returns:
            torch.device instance (mps or cpu)
        """
        if self._device is None:
            self._device = self._select_device()
        return self._device

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon hardware.

        Returns:
            True if running on Apple Silicon, False otherwise
        """
        try:
            # Check platform
            if platform.system() != "Darwin":
                logger.debug("Not running on macOS")
                return False

            # Check architecture
            machine = platform.machine().lower()
            if machine != "arm64":
                logger.debug(f"Not running on ARM64 architecture (detected: {machine})")
                return False

            # Verify Apple Silicon via sysctl
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            cpu_brand = result.stdout.strip()

            is_apple_silicon = "Apple" in cpu_brand
            if is_apple_silicon:
                logger.info(f"Detected Apple Silicon: {cpu_brand}")
            else:
                logger.debug(f"Not Apple Silicon (CPU: {cpu_brand})")

            return is_apple_silicon

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Could not detect Apple Silicon: {e}")
            return False

    def _check_mps_availability(self) -> bool:
        """Check if MPS backend is available in PyTorch.

        Returns:
            True if MPS is available, False otherwise
        """
        try:
            import torch

            if not hasattr(torch.backends, "mps"):
                logger.debug("PyTorch MPS backend not available")
                return False

            is_available = torch.backends.mps.is_available()
            if is_available:
                logger.info("PyTorch MPS backend is available")
            else:
                logger.debug("PyTorch MPS backend not available")

            return is_available

        except (ImportError, AttributeError) as e:
            logger.debug(f"Error checking MPS availability: {e}")
            return False

    def _select_device(self) -> torch.device:
        """Select the appropriate device for computation.

        Returns:
            torch.device instance (mps or cpu)
        """
        import torch

        # If user disabled MPS, use CPU
        if not self.use_mps:
            logger.info("MPS disabled by configuration, using CPU")
            return torch.device("cpu")

        # Check if MPS is available
        if self.is_apple_silicon and self.is_mps_available:
            logger.info("Using MPS device for GPU acceleration")
            return torch.device("mps")

        # Fallback to CPU
        if self.is_apple_silicon:
            logger.warning("Apple Silicon detected but MPS not available, falling back to CPU")
        else:
            logger.info("Not running on Apple Silicon, using CPU")

        return torch.device("cpu")

    def log_device_info(self) -> None:
        """Log detailed device information."""
        logger.info("=" * 50)
        logger.info("Device Configuration")
        logger.info("=" * 50)
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Architecture: {platform.machine()}")
        logger.info(f"Apple Silicon: {self.is_apple_silicon}")
        logger.info(f"MPS Available: {self.is_mps_available}")
        logger.info(f"Selected Device: {self.device}")
        logger.info("=" * 50)

    def get_memory_info(self) -> dict[str, object]:
        """Get memory information for the device.

        Returns:
            Dictionary containing memory information
        """
        import torch

        memory_info: dict[str, object] = {
            "device": str(self.device),
            "device_type": self.device.type,
        }

        # MPS doesn't have direct memory query APIs yet
        if self.device.type == "mps":
            memory_info["note"] = "MPS uses unified memory architecture"
            memory_info["available"] = "shared with system RAM"
        else:
            memory_info["note"] = "Running on CPU"

        return memory_info

    def optimize_for_device(self) -> None:
        """Apply device-specific optimizations."""
        if self.device.type == "mps":
            logger.info("Applying MPS-specific optimizations")
            # MPS-specific optimizations can be added here
            # For now, just log that we're using MPS
        else:
            logger.info("No device-specific optimizations for CPU")
