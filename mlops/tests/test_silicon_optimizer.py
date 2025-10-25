"""Tests for Apple Silicon Optimizer"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mlops.silicon.detector import HardwareInfo
from mlops.silicon.optimizer import AppleSiliconOptimizer, OptimalConfig


class TestAppleSiliconOptimizer:
    """Test suite for AppleSiliconOptimizer"""

    @pytest.fixture
    def m1_hardware_info(self):
        """Mock M1 hardware info"""
        return HardwareInfo(
            is_apple_silicon=True,
            chip_type="M1",
            chip_variant="Base",
            system="Darwin",
            machine="arm64",
            processor="arm",
            memory_total_gb=16.0,
            core_count=8,
            performance_cores=4,
            efficiency_cores=4,
            mlx_available=True,
            mps_available=True,
            ane_available=True,
            thermal_state=0,
            power_mode="normal",
        )

    @pytest.fixture
    def m2_pro_hardware_info(self):
        """Mock M2 Pro hardware info"""
        return HardwareInfo(
            is_apple_silicon=True,
            chip_type="M2",
            chip_variant="Pro",
            system="Darwin",
            machine="arm64",
            processor="arm",
            memory_total_gb=32.0,
            core_count=12,
            performance_cores=8,
            efficiency_cores=4,
            mlx_available=True,
            mps_available=True,
            ane_available=True,
            thermal_state=0,
            power_mode="high_performance",
        )

    @pytest.fixture
    def m3_max_hardware_info(self):
        """Mock M3 Max hardware info"""
        return HardwareInfo(
            is_apple_silicon=True,
            chip_type="M3",
            chip_variant="Max",
            system="Darwin",
            machine="arm64",
            processor="arm",
            memory_total_gb=64.0,
            core_count=16,
            performance_cores=12,
            efficiency_cores=4,
            mlx_available=True,
            mps_available=True,
            ane_available=True,
            thermal_state=0,
            power_mode="high_performance",
        )

    @pytest.fixture
    def non_apple_silicon_info(self):
        """Mock non-Apple Silicon hardware info"""
        return HardwareInfo(
            is_apple_silicon=False,
            chip_type="x86_64",
            chip_variant="Unknown",
            system="Linux",
            machine="x86_64",
            processor="x86_64",
            memory_total_gb=32.0,
            core_count=8,
            performance_cores=8,
            efficiency_cores=0,
            mlx_available=False,
            mps_available=False,
            ane_available=False,
            thermal_state=0,
            power_mode="normal",
        )

    def test_optimizer_initialization(self, m1_hardware_info):
        """Test optimizer initializes correctly"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        assert optimizer.hardware_info == m1_hardware_info

    def test_get_optimal_config_inference(self, m1_hardware_info):
        """Test optimal config for inference workload"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config(workload_type="inference")

        assert isinstance(config, OptimalConfig)
        assert config.workers >= 1
        assert config.batch_size >= 1
        assert config.memory_limit_gb > 0
        assert config.use_mlx is True
        assert len(config.recommendations) > 0

    def test_get_optimal_config_training(self, m1_hardware_info):
        """Test optimal config for training workload"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config(workload_type="training")

        assert isinstance(config, OptimalConfig)
        assert config.workers <= 2  # Conservative for training
        assert config.prefetch_batches == 3  # More for training

    def test_get_optimal_config_serving(self, m2_pro_hardware_info):
        """Test optimal config for serving workload"""
        optimizer = AppleSiliconOptimizer(m2_pro_hardware_info)
        config = optimizer.get_optimal_config(workload_type="serving")

        assert isinstance(config, OptimalConfig)
        assert config.workers >= 2  # More workers for serving
        assert config.batch_size > 0

    def test_memory_intensive_config(self, m1_hardware_info):
        """Test config with memory intensive workload"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config(
            workload_type="training",
            memory_intensive=True,
        )

        # Should have smaller batch size for memory-intensive workload
        assert config.batch_size <= 32

    def test_non_memory_intensive_config(self, m3_max_hardware_info):
        """Test config with non-memory intensive workload"""
        optimizer = AppleSiliconOptimizer(m3_max_hardware_info)
        config = optimizer.get_optimal_config(
            workload_type="inference",
            memory_intensive=False,
        )

        # Should have larger batch size with more memory
        assert config.batch_size >= 64

    def test_thermal_throttling_adjustment(self, m1_hardware_info):
        """Test config adjusts for thermal throttling"""
        # Set thermal state to serious
        m1_hardware_info.thermal_state = 2

        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()

        # Should reduce workers due to thermal throttling
        assert config.workers <= 2
        assert any("thermal throttling" in r.lower() for r in config.recommendations)

    def test_low_power_mode_adjustment(self, m1_hardware_info):
        """Test config adjusts for low power mode"""
        m1_hardware_info.power_mode = "low_power"

        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()

        # Should reduce workers and batch size
        assert config.workers <= 2
        assert any("low power" in r.lower() for r in config.recommendations)

    def test_memory_limit_calculation(self, m1_hardware_info):
        """Test memory limit is 80% of total"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()

        expected_limit = m1_hardware_info.memory_total_gb * 0.8
        assert config.memory_limit_gb == pytest.approx(expected_limit, rel=0.01)

    def test_batch_size_scaling(self):
        """Test batch size scales with memory"""
        # Test different memory configurations
        memory_configs = [
            (16.0, 16, 32),   # 16GB: 16-32 batch size
            (32.0, 32, 64),   # 32GB: 32-64 batch size
            (64.0, 64, 128),  # 64GB: 64-128 batch size
        ]

        for memory_gb, min_batch, max_batch in memory_configs:
            info = HardwareInfo(
                is_apple_silicon=True,
                chip_type="M2",
                chip_variant="Base",
                system="Darwin",
                machine="arm64",
                processor="arm",
                memory_total_gb=memory_gb,
                core_count=8,
                performance_cores=4,
                efficiency_cores=4,
                mlx_available=True,
                mps_available=True,
                ane_available=True,
                thermal_state=0,
                power_mode="normal",
            )

            optimizer = AppleSiliconOptimizer(info)
            config = optimizer.get_optimal_config()

            assert min_batch <= config.batch_size <= max_batch

    def test_worker_count_scaling(self, m2_pro_hardware_info):
        """Test worker count scales with performance cores"""
        optimizer = AppleSiliconOptimizer(m2_pro_hardware_info)

        # Test different workload types
        inference_config = optimizer.get_optimal_config(workload_type="inference")
        serving_config = optimizer.get_optimal_config(workload_type="serving")

        # Serving should have more workers than inference
        assert serving_config.workers >= inference_config.workers

    def test_mlx_preference_over_mps(self, m1_hardware_info):
        """Test MLX is preferred over MPS when both available"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()

        assert config.use_mlx is True
        assert config.use_mps is False  # Should prefer MLX

    def test_mps_fallback(self, m1_hardware_info):
        """Test MPS is used when MLX not available"""
        m1_hardware_info.mlx_available = False
        m1_hardware_info.mps_available = True

        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()

        assert config.use_mlx is False
        assert config.use_mps is True

    def test_non_apple_silicon_fallback(self, non_apple_silicon_info):
        """Test fallback config for non-Apple Silicon"""
        optimizer = AppleSiliconOptimizer(non_apple_silicon_info)
        config = optimizer.get_optimal_config()

        assert config.workers == 1
        assert config.batch_size == 32
        assert config.use_mlx is False
        assert config.use_mps is False
        assert "Not running on Apple Silicon" in config.recommendations[0]

    def test_get_deployment_config(self, m1_hardware_info):
        """Test deployment config generation"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_deployment_config()

        assert isinstance(config, dict)
        assert "workers" in config
        assert "max_batch_size" in config
        assert "memory_per_worker_mb" in config
        assert "apple_silicon" in config
        assert config["apple_silicon"]["enabled"] is True

    def test_get_training_config(self, m1_hardware_info):
        """Test training config generation"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_training_config()

        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_workers" in config
        assert "use_mlx" in config
        assert "memory_limit_gb" in config
        assert config["persistent_workers"] is True
        assert config["pin_memory"] is False  # Not needed on Apple Silicon

    def test_optimal_config_to_dict(self, m1_hardware_info):
        """Test OptimalConfig to_dict conversion"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "workers" in config_dict
        assert "batch_size" in config_dict
        assert "recommendations" in config_dict
        assert isinstance(config_dict["recommendations"], list)

    def test_recommendations_content(self, m1_hardware_info):
        """Test recommendations contain useful information"""
        optimizer = AppleSiliconOptimizer(m1_hardware_info)
        config = optimizer.get_optimal_config(workload_type="inference")

        # Should have multiple recommendations
        assert len(config.recommendations) >= 3

        # Should mention workers
        assert any("worker" in r.lower() for r in config.recommendations)

        # Should mention batch size
        assert any("batch" in r.lower() for r in config.recommendations)

        # Should mention memory
        assert any("memory" in r.lower() for r in config.recommendations)
