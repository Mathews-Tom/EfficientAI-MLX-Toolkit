"""Training callbacks for style transfer models."""

from typing import Any, Dict


class TrainingCallbacks:
    """Callbacks for handling training events."""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, metrics)
    
    def on_training_start(self) -> None:
        """Called at the start of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start()
    
    def on_training_end(self) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end()


class EarlyStoppingCallback:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Return True if training should stop."""
        loss = metrics.get('loss', float('inf'))
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        return self.wait >= self.patience


class ModelCheckpointCallback:
    """Model checkpointing callback."""
    
    def __init__(self, filepath: str, save_best_only: bool = True):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint if conditions are met."""
        loss = metrics.get('loss', float('inf'))
        
        if not self.save_best_only or loss < self.best_loss:
            if loss < self.best_loss:
                self.best_loss = loss
            # In a real implementation, this would save the model
            print(f"Saving model checkpoint at epoch {epoch}")