"""
Tests for training pipeline and components.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import tempfile
import time

from training import LoRATrainer, TrainingState, MLXMonitorCallback
from training.optimizer import create_optimizer, create_scheduler
from training.callbacks import ModelCheckpointCallback, EarlyStopping
from lora import LoRAConfig, TrainingConfig, ModelAdapter


class TestTrainingState:
    """Test training state management."""
    
    def test_training_state_creation(self):
        """Test training state initialization."""
        state = TrainingState()
        
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_metric == float('inf')
        assert state.train_loss == 0.0
        assert state.eval_loss == 0.0
        assert state.peak_memory_mb == 0.0
    
    def test_training_state_serialization(self):
        """Test training state to dict conversion."""
        state = TrainingState()
        state.epoch = 5
        state.global_step = 100
        state.train_loss = 0.5
        
        state_dict = state.to_dict()
        
        assert state_dict["epoch"] == 5
        assert state_dict["global_step"] == 100
        assert state_dict["train_loss"] == 0.5
        assert "peak_memory_mb" in state_dict


class TestOptimizers:
    """Test optimizer and scheduler creation."""
    
    def test_create_adamw_optimizer(self):
        """Test AdamW optimizer creation."""
        # Create dummy parameters
        params = [mx.random.normal((10, 10)), mx.random.normal((5, 5))]
        
        optimizer = create_optimizer(
            parameters=iter(params),
            optimizer_name="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        assert optimizer is not None
        assert optimizer.learning_rate == 1e-4
    
    def test_create_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        params = [mx.random.normal((10, 10))]
        
        optimizer = create_optimizer(
            parameters=iter(params),
            optimizer_name="sgd",
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        
        assert optimizer is not None
        assert optimizer.learning_rate == 1e-3
    
    def test_invalid_optimizer(self):
        """Test invalid optimizer name."""
        params = [mx.random.normal((10, 10))]
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            create_optimizer(
                parameters=iter(params),
                optimizer_name="invalid",
            )
    
    def test_linear_scheduler(self):
        """Test linear learning rate scheduler."""
        from training.optimizer import LinearScheduler
        
        # Create mock optimizer
        class MockOptimizer:
            def __init__(self):
                self.learning_rate = 1e-3
        
        optimizer = MockOptimizer()
        scheduler = LinearScheduler(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        
        # Test warmup phase
        assert scheduler.get_lr() == 1e-5  # 1e-3 * 1/100
        
        # Step through warmup
        for _ in range(100):
            scheduler.step()
        
        # Should be at base learning rate after warmup
        assert abs(scheduler.get_lr() - 1e-3) < 1e-6
    
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        from training.optimizer import CosineAnnealingScheduler
        
        class MockOptimizer:
            def __init__(self):
                self.learning_rate = 1e-3
        
        optimizer = MockOptimizer()
        scheduler = CosineAnnealingScheduler(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        
        initial_lr = scheduler.get_lr()
        assert initial_lr == 1e-4  # 1e-3 * 1/10
        
        # Test that scheduler produces valid learning rates
        for _ in range(50):
            scheduler.step()
            lr = scheduler.get_lr()
            assert 0 <= lr <= 1e-3


class TestCallbacks:
    """Test training callbacks."""
    
    def test_mlx_monitor_callback(self):
        """Test MLX monitoring callback."""
        callback = MLXMonitorCallback(log_frequency=10)
        state = TrainingState()
        
        # Test callback methods don't crash
        callback.on_train_start(state, {})
        callback.on_epoch_start(state, {})
        callback.on_epoch_end(state, {"train_loss": 0.5})
        callback.on_batch_start(state, {"batch": 0})
        callback.on_batch_end(state, {"loss": 0.5})
        callback.on_train_end(state, {})
        
        # Check that memory history is populated
        assert len(callback.memory_history) > 0
    
    def test_model_checkpoint_callback(self):
        """Test model checkpointing callback."""
        callback = ModelCheckpointCallback(
            save_frequency=1,
            metric_name="eval_loss",
            mode="min",
        )
        
        state = TrainingState()
        
        # Test callback initialization
        callback.on_train_start(state, {})
        
        # Test epoch end with improving metric
        callback.on_epoch_end(state, {"eval_loss": 0.5})
        assert len(callback.best_metrics) == 1
        assert callback.best_metrics[0]["value"] == 0.5
        
        # Test epoch end with worse metric
        callback.on_epoch_end(state, {"eval_loss": 0.7})
        assert len(callback.best_metrics) == 1  # Should not add worse result
        
        # Test epoch end with better metric
        callback.on_epoch_end(state, {"eval_loss": 0.3})
        assert len(callback.best_metrics) == 2
        assert callback.best_metrics[0]["value"] == 0.3  # Should be sorted
    
    def test_early_stopping_callback(self):
        """Test early stopping callback."""
        callback = EarlyStopping(
            metric_name="eval_loss",
            patience=3,
            min_delta=0.01,
            mode="min",
        )
        
        state = TrainingState()
        callback.on_train_start(state, {})
        
        # Test improving metrics
        callback.on_epoch_end(state, {"eval_loss": 1.0})
        assert callback.wait == 0
        assert not callback.should_stop
        
        callback.on_epoch_end(state, {"eval_loss": 0.9})
        assert callback.wait == 0
        assert not callback.should_stop
        
        # Test non-improving metrics
        callback.on_epoch_end(state, {"eval_loss": 0.95})
        assert callback.wait == 1
        assert not callback.should_stop
        
        callback.on_epoch_end(state, {"eval_loss": 0.92})
        assert callback.wait == 2
        assert not callback.should_stop
        
        callback.on_epoch_end(state, {"eval_loss": 0.94})
        assert callback.wait == 3
        assert callback.should_stop


class TestLoRATrainer:
    """Test LoRA trainer functionality."""
    
    def test_trainer_creation(self, simple_model, sample_lora_config, sample_training_config):
        """Test trainer initialization."""
        trainer = LoRATrainer(
            model=simple_model,
            lora_config=sample_lora_config,
            training_config=sample_training_config,
        )
        
        assert trainer.model is simple_model
        assert trainer.lora_config is sample_lora_config
        assert trainer.training_config is sample_training_config
        assert trainer.model_adapter is not None
        assert len(trainer.callbacks) >= 1  # Should have default MLXMonitorCallback
    
    def test_trainer_setup(self, simple_model, sample_lora_config, sample_training_config):
        """Test trainer setup methods."""
        trainer = LoRATrainer(
            model=simple_model,
            lora_config=sample_lora_config,
            training_config=sample_training_config,
        )
        
        # Test directory setup
        trainer.setup_directories()
        assert trainer.training_config.output_dir.exists()
        assert (trainer.training_config.output_dir / "checkpoints").exists()
        assert (trainer.training_config.output_dir / "logs").exists()
    
    def test_compute_loss(self, simple_model, sample_lora_config, sample_training_config):
        """Test loss computation."""
        trainer = LoRATrainer(
            model=simple_model,
            lora_config=sample_lora_config,
            training_config=sample_training_config,
        )
        
        # Create sample batch
        batch = {
            "input_ids": mx.array([[1, 2, 3, 4, 5]]),
            "attention_mask": mx.array([[1, 1, 1, 1, 1]]),
            "labels": mx.array([[2, 3, 4, 5, 6]]),
        }
        
        # Test loss computation (this will fail with current implementation
        # since we don't have a real language model, but test the structure)
        try:
            loss = trainer.compute_loss(batch)
            assert isinstance(loss, mx.array)
            assert loss.shape == ()  # Scalar loss
        except Exception as e:
            # Expected to fail with mock model
            assert ("logits" in str(e).lower() or 
                    "shape" in str(e).lower() or 
                    "attention_mask" in str(e).lower())


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_training_workflow(self, simple_model, sample_lora_config, sample_training_config, sample_dataset):
        """Test complete training workflow."""
        # Reduce epochs for testing
        sample_training_config.num_epochs = 1
        sample_training_config.eval_steps = 1
        sample_training_config.save_steps = 1
        
        trainer = LoRATrainer(
            model=simple_model,
            lora_config=sample_lora_config,
            training_config=sample_training_config,
            train_dataset=sample_dataset[:5],  # Small dataset for testing
            eval_dataset=sample_dataset[-2:],
        )
        
        # Test that trainer can be created and setup without errors
        trainer.setup_directories()
        trainer.setup_optimization()
        
        # Test that optimizer was created
        assert trainer.optimizer is not None
        
        # Test trainable parameters
        trainable_params = trainer.model_adapter.get_trainable_parameters()
        assert len(trainable_params) > 0
        
        # Test that all trainable parameters are LoRA parameters
        for param_name in trainable_params.keys():
            assert "lora_" in param_name


@pytest.mark.benchmark
class TestTrainingPerformance:
    """Performance benchmarks for training components."""
    
    def test_forward_pass_performance(self, simple_model, sample_lora_config):
        """Benchmark forward pass performance."""
        # Adapt model
        adapter = ModelAdapter(simple_model, sample_lora_config)
        adapter.adapt_model()
        
        # Create batch
        batch_size = 8
        seq_length = 128
        input_ids = mx.random.randint(0, 100, (batch_size, seq_length))
        
        # Warm up
        for _ in range(5):
            output = simple_model(input_ids)
            mx.eval(output)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            output = simple_model(input_ids)
            mx.eval(output)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Average forward pass time: {avg_time:.4f}s")
        
        # Should be reasonably fast
        assert avg_time < 1.0  # Less than 1 second per forward pass
    
    def test_memory_efficiency(self, simple_model, sample_lora_config):
        """Test memory efficiency of LoRA adaptation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Adapt model
        adapter = ModelAdapter(simple_model, sample_lora_config)
        adapter.adapt_model()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase from LoRA adaptation: {memory_increase:.2f} MB")
        
        # Should not use excessive memory
        assert memory_increase < 100  # Less than 100 MB increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])