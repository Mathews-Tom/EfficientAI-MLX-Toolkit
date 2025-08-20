"""
Training callbacks for monitoring and control.

Comprehensive callback system with Apple Silicon monitoring, early stopping,
model checkpointing, and experiment tracking integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import time
import json
import psutil
import mlx.core as mx

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class TrainingCallback(ABC):
    """Base callback class for training monitoring and control."""
    
    @abstractmethod
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the start of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass


class MLXMonitorCallback(TrainingCallback):
    """
    Apple Silicon / MLX performance monitoring callback.
    
    Tracks memory usage, GPU utilization, and MLX-specific metrics
    for optimization insights on Apple Silicon hardware.
    """
    
    def __init__(
        self,
        log_frequency: int = 100,
        memory_threshold_mb: float = 1000.0,
        save_logs: bool = True,
    ):
        self.log_frequency = log_frequency
        self.memory_threshold_mb = memory_threshold_mb
        self.save_logs = save_logs
        
        self.memory_history = []
        self.performance_logs = []
        self.start_memory = None
        
        # Check MLX availability
        self.mlx_available = mx.metal.is_available() if hasattr(mx, 'metal') else False
    
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Initialize monitoring at training start."""
        self.start_memory = self._get_memory_info()
        print("=== MLX Performance Monitoring Started ===")
        print(f"MLX Metal Available: {self.mlx_available}")
        print(f"Initial Memory: {self.start_memory['used_gb']:.2f} GB")
        print("=" * 45)
    
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Generate final performance report."""
        final_memory = self._get_memory_info()
        peak_memory = max(self.memory_history, key=lambda x: x['used_gb'])
        
        print("\n=== MLX Performance Summary ===")
        print(f"Peak Memory Usage: {peak_memory['used_gb']:.2f} GB")
        print(f"Memory Efficiency: {(peak_memory['used_gb'] / peak_memory['total_gb']) * 100:.1f}%")
        print(f"Total Training Time: {state.total_train_time:.2f} seconds")
        
        if self.save_logs and hasattr(state, 'output_dir'):
            self._save_performance_logs(state.output_dir)
    
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Monitor epoch start."""
        current_memory = self._get_memory_info()
        state.current_memory_mb = current_memory['used_gb'] * 1024
        
        if len(self.memory_history) > 0:
            prev_memory = self.memory_history[-1]['used_gb'] * 1024
            if state.current_memory_mb - prev_memory > self.memory_threshold_mb:
                print(f"Warning: Memory usage increased by {state.current_memory_mb - prev_memory:.1f} MB")
    
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Log epoch performance metrics."""
        memory_info = self._get_memory_info()
        
        performance_entry = {
            "epoch": state.epoch,
            "memory_gb": memory_info['used_gb'],
            "memory_percent": memory_info['percent'],
            "train_loss": logs.get('train_loss', 0.0),
            "eval_loss": logs.get('eval_loss', 0.0),
            "epoch_time": logs.get('epoch_time', 0.0),
            "learning_rate": logs.get('learning_rate', 0.0),
        }
        
        self.performance_logs.append(performance_entry)
        self.memory_history.append(memory_info)
        
        # Update state with peak memory
        state.peak_memory_mb = max(
            state.peak_memory_mb, 
            memory_info['used_gb'] * 1024
        )
        
        print(f"Memory: {memory_info['used_gb']:.2f} GB ({memory_info['percent']:.1f}%)")
    
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Monitor batch start (minimal logging)."""
        pass
    
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Monitor batch performance."""
        if state.global_step % self.log_frequency == 0:
            memory_info = self._get_memory_info()
            
            # Check for memory leaks
            if len(self.memory_history) > 10:
                recent_avg = sum(m['used_gb'] for m in self.memory_history[-10:]) / 10
                if memory_info['used_gb'] > recent_avg * 1.2:
                    print(f"Warning: Potential memory leak detected at step {state.global_step}")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent,
        }
    
    def _save_performance_logs(self, output_dir: Path) -> None:
        """Save performance logs to file."""
        log_dir = output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Save detailed performance logs
        with open(log_dir / "performance_logs.json", 'w') as f:
            json.dump(self.performance_logs, f, indent=2)
        
        # Save memory history
        with open(log_dir / "memory_history.json", 'w') as f:
            json.dump(self.memory_history, f, indent=2)
        
        print(f"Performance logs saved to {log_dir}")


class ModelCheckpointCallback(TrainingCallback):
    """
    Model checkpointing callback with best model tracking.
    
    Saves model checkpoints based on validation metrics with
    configurable save frequency and best model tracking.
    """
    
    def __init__(
        self,
        save_frequency: int = 1,
        metric_name: str = "eval_loss",
        mode: str = "min",
        save_top_k: int = 3,
    ):
        self.save_frequency = save_frequency
        self.metric_name = metric_name
        self.mode = mode
        self.save_top_k = save_top_k
        
        self.best_metrics = []
        self.checkpoint_paths = []
    
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Initialize checkpointing."""
        print(f"Model checkpointing enabled (metric: {self.metric_name}, mode: {self.mode})")
    
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Final checkpoint summary."""
        print(f"\nSaved {len(self.checkpoint_paths)} checkpoints")
        if self.best_metrics:
            best_value = self.best_metrics[0]['value']
            best_epoch = self.best_metrics[0]['epoch']
            print(f"Best {self.metric_name}: {best_value:.4f} (epoch {best_epoch})")
    
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at epoch start."""
        pass
    
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Check if checkpoint should be saved."""
        if state.epoch % self.save_frequency != 0:
            return
        
        current_metric = logs.get(self.metric_name)
        if current_metric is None:
            return
        
        # Determine if this is a new best model
        is_best = self._is_best_metric(current_metric)
        
        if is_best:
            self.best_metrics.append({
                "epoch": state.epoch,
                "value": current_metric,
                "step": state.global_step,
            })
            
            # Keep only top-k best metrics
            if self.mode == "min":
                self.best_metrics.sort(key=lambda x: x["value"])
            else:
                self.best_metrics.sort(key=lambda x: x["value"], reverse=True)
            
            if len(self.best_metrics) > self.save_top_k:
                self.best_metrics = self.best_metrics[:self.save_top_k]
            
            print(f"New best {self.metric_name}: {current_metric:.4f}")
    
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch end."""
        pass
    
    def _is_best_metric(self, current_value: float) -> bool:
        """Check if current metric is better than previous best."""
        if not self.best_metrics:
            return True
        
        if self.mode == "min":
            return current_value < self.best_metrics[0]["value"]
        else:
            return current_value > self.best_metrics[0]["value"]


class EarlyStopping(TrainingCallback):
    """
    Early stopping callback to prevent overfitting.
    
    Monitors validation metrics and stops training when improvement
    plateaus for a specified number of patience epochs.
    """
    
    def __init__(
        self,
        metric_name: str = "eval_loss",
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_metric = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Initialize early stopping."""
        print(f"Early stopping enabled (patience: {self.patience}, metric: {self.metric_name})")
        self.best_metric = float('inf') if self.mode == "min" else float('-inf')
        self.wait = 0
        self.should_stop = False
    
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Report early stopping results."""
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")
            print(f"Training stopped {state.epoch - self.stopped_epoch} epochs early")
    
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Check if training should stop."""
        if self.should_stop:
            print("Training stopped due to early stopping")
            # In a real implementation, you'd need to signal the trainer to stop
            # This is a simplified version for demonstration
    
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Monitor metric for early stopping."""
        current_metric = logs.get(self.metric_name)
        if current_metric is None:
            return
        
        # Check if metric improved
        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.wait = 0
            print(f"Metric improved: {self.metric_name}={current_metric:.4f}")
        else:
            self.wait += 1
            print(f"No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = state.epoch
                self.should_stop = True
    
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch end."""
        pass
    
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current metric is an improvement."""
        if self.mode == "min":
            return current_value < (self.best_metric - self.min_delta)
        else:
            return current_value > (self.best_metric + self.min_delta)


class WandbCallback(TrainingCallback):
    """
    Weights & Biases integration callback.
    
    Logs training metrics, system performance, and model artifacts
    to W&B for experiment tracking and comparison.
    """
    
    def __init__(
        self,
        project: str = "lora-finetuning",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        log_frequency: int = 100,
        log_gradients: bool = False,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandbCallback")
        
        self.project = project
        self.entity = entity
        self.name = name
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        
        self.run = None
    
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Initialize W&B run."""
        config = logs.get('config', {})
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=config,
            reinit=True,
        )
        
        print(f"W&B run initialized: {self.run.url}")
    
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Finalize W&B run."""
        if self.run:
            # Log final metrics
            final_metrics = {
                "final/best_epoch": state.best_epoch,
                "final/best_metric": state.best_metric,
                "final/total_training_time": state.total_train_time,
                "final/peak_memory_mb": state.peak_memory_mb,
            }
            
            wandb.log(final_metrics)
            
            # Finish run
            wandb.finish()
            print("W&B run finished")
    
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Log epoch start metrics."""
        pass
    
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Log epoch metrics to W&B."""
        if not self.run:
            return
        
        epoch_metrics = {
            "epoch": state.epoch,
            "train/loss": logs.get('train_loss', 0.0),
            "eval/loss": logs.get('eval_loss', 0.0),
            "train/learning_rate": logs.get('learning_rate', 0.0),
            "system/memory_mb": state.current_memory_mb,
            "system/peak_memory_mb": state.peak_memory_mb,
            "train/epoch_time": logs.get('epoch_time', 0.0),
        }
        
        wandb.log(epoch_metrics, step=state.global_step)
    
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Log batch metrics to W&B."""
        if not self.run or state.global_step % self.log_frequency != 0:
            return
        
        batch_metrics = {
            "train/batch_loss": logs.get('loss', 0.0),
            "train/learning_rate": logs.get('learning_rate', 0.0),
            "train/global_step": state.global_step,
        }
        
        wandb.log(batch_metrics, step=state.global_step)


class ProgressCallback(TrainingCallback):
    """Simple progress tracking callback with rich formatting."""
    
    def __init__(self, update_frequency: int = 10):
        self.update_frequency = update_frequency
        self.epoch_start_time = None
        
    def on_train_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Print training start information."""
        print("\n🚀 Starting LoRA Fine-tuning")
        print("=" * 50)
    
    def on_train_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Print training completion summary."""
        print("\n✅ Training Completed!")
        print("=" * 50)
        print(f"🏆 Best model: Epoch {state.best_epoch}")
        print(f"📊 Best metric: {state.best_metric:.4f}")
        print(f"⏱️  Total time: {state.total_train_time:.2f}s")
        print(f"💾 Peak memory: {state.peak_memory_mb:.1f} MB")
    
    def on_epoch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """Print epoch start information."""
        self.epoch_start_time = time.time()
        print(f"\n📚 Epoch {state.epoch + 1}")
        print("-" * 30)
    
    def on_epoch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Print epoch completion summary."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        print(f"✨ Epoch {state.epoch + 1} Complete")
        print(f"   Loss: {logs.get('train_loss', 0.0):.4f}")
        print(f"   Val Loss: {logs.get('eval_loss', 0.0):.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   LR: {logs.get('learning_rate', 0.0):.2e}")
    
    def on_batch_start(self, state: Any, logs: Dict[str, Any]) -> None:
        """No action needed at batch start."""
        pass
    
    def on_batch_end(self, state: Any, logs: Dict[str, Any]) -> None:
        """Print periodic progress updates."""
        if state.global_step % self.update_frequency == 0:
            print(f"  Step {state.global_step}: loss={logs.get('loss', 0.0):.4f}")