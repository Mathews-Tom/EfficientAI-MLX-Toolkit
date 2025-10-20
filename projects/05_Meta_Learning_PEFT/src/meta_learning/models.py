"""Neural network models for meta-learning experiments."""

import mlx.core as mx
import mlx.nn as nn


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for few-shot learning."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2
    ):
        """Initialize classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of output classes.
            num_layers: Number of hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Build layers as named modules for proper gradient structure
        in_dim = input_dim
        for i in range(num_layers):
            setattr(self, f"hidden_{i}", nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.output = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Apply hidden layers with ReLU activation
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = nn.relu(x)

        # Output layer (no activation)
        x = self.output(x)
        return x


class LinearClassifier(nn.Module):
    """Simple linear classifier for baseline comparison."""

    def __init__(self, input_dim: int, num_classes: int):
        """Initialize linear classifier.

        Args:
            input_dim: Input feature dimension.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.linear(x)


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss for classification.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes).
        targets: Target labels of shape (batch_size,).

    Returns:
        Scalar loss.
    """
    # Compute log softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Select log probabilities for target classes
    num_classes = logits.shape[-1]
    targets_one_hot = mx.eye(num_classes)[targets]
    selected_log_probs = mx.sum(log_probs * targets_one_hot, axis=-1)

    # Return mean negative log likelihood
    return -mx.mean(selected_log_probs)


def accuracy(logits: mx.array, targets: mx.array) -> mx.array:
    """Compute classification accuracy.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes).
        targets: Target labels of shape (batch_size,).

    Returns:
        Scalar accuracy in [0, 1].
    """
    preds = mx.argmax(logits, axis=-1)
    return mx.mean((preds == targets).astype(mx.float32))
