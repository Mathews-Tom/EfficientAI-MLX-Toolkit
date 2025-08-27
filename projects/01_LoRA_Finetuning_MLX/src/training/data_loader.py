"""
MLX-optimized data loading for LoRA fine-tuning.

Handles JSONL conversation data, tokenization, and batch creation
with efficient MLX tensor operations and Apple Silicon optimization.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import shared utilities with fallback
try:
    from utils.logging_utils import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


from datasets import Dataset
from transformers import AutoTokenizer

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    add_special_tokens: bool = True
    chat_template: str | None = None
    ignore_index: int = -100


@dataclass
class ConversationExample:
    """Container for a single conversation example."""

    prompt: str
    completion: str
    input_ids: mx.array | None = None
    attention_mask: mx.array | None = None
    labels: mx.array | None = None


class ConversationDataLoader:
    """
    MLX-optimized data loader for conversation data.

    Loads JSONL files with conversation data and creates MLX-compatible
    batches for training with efficient memory usage and Apple Silicon optimization.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: DatasetConfig | None = None,
    ):
        """
        Initialize data loader.

        Args:
            tokenizer: Tokenizer for text processing
            config: Dataset configuration
        """
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()

    def load_jsonl(self, file_path: str | Path) -> list[ConversationExample]:
        """
        Load conversations from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of conversation examples
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        conversations = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract prompt and completion
                    if "prompt" in data and "completion" in data:
                        conversation = ConversationExample(
                            prompt=data["prompt"], completion=data["completion"]
                        )
                        conversations.append(conversation)
                    else:
                        logger.warning(
                            "Skipping line %d, missing 'prompt' or 'completion'", line_num
                        )

                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON on line %d: %s", line_num, e)
                    continue

        logger.info("Loaded %d conversations from %s", len(conversations), file_path)
        return conversations

    def _format_conversation_text(self, conversation: ConversationExample) -> str:
        """Format conversation prompt and completion into input text."""
        if self.config.chat_template:
            return self.config.chat_template.format(
                prompt=conversation.prompt, completion=conversation.completion
            )
        else:
            return f"{conversation.prompt} {conversation.completion}"

    def _is_mlx_tokenizer(self) -> bool:
        """Check if tokenizer is MLX-style (has encode but not __call__)."""
        return hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "__call__")

    def _tokenize_with_mlx(self, input_text: str, conversation: ConversationExample) -> tuple[list, list, list]:
        """Tokenize using MLX-style tokenizer."""
        input_ids = self.tokenizer.encode(input_text)
        prompt_ids = self.tokenizer.encode(conversation.prompt)

        # Truncate if necessary
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[: self.config.max_length]

        # Pad if necessary
        if self.config.padding and len(input_ids) < self.config.max_length:
            pad_length = self.config.max_length - len(input_ids)
            input_ids = input_ids + [0] * pad_length  # Assuming 0 is pad token

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * min(len(input_ids), self.config.max_length)
        if len(attention_mask) < self.config.max_length:
            attention_mask += [0] * (self.config.max_length - len(attention_mask))

        # Create labels - mask prompt tokens with ignore_index
        labels = input_ids.copy()
        prompt_length = len(prompt_ids)
        if prompt_length < len(labels):
            labels[:prompt_length] = [self.config.ignore_index] * prompt_length

        return input_ids, attention_mask, labels

    def _tokenize_with_huggingface(self, input_text: str, conversation: ConversationExample) -> tuple[list, list, list]:
        """Tokenize using HuggingFace-style tokenizer."""
        encoding = self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            padding="max_length" if self.config.padding else False,
            truncation=self.config.truncation,
            add_special_tokens=self.config.add_special_tokens,
            return_tensors="np",
        )

        # Create labels for causal language modeling
        prompt_encoding = self.tokenizer(
            conversation.prompt,
            add_special_tokens=self.config.add_special_tokens,
            return_tensors="np",
        )
        prompt_length = len(prompt_encoding["input_ids"][0])

        # Create labels - mask prompt tokens with ignore_index
        labels = encoding["input_ids"].copy()
        labels[0, :prompt_length] = self.config.ignore_index

        return encoding["input_ids"][0], encoding["attention_mask"][0], labels[0]

    def tokenize_conversation(self, conversation: ConversationExample) -> ConversationExample:
        """
        Tokenize a single conversation example.

        Args:
            conversation: Raw conversation example

        Returns:
            Tokenized conversation example
        """
        # Format conversation text
        input_text = self._format_conversation_text(conversation)

        # Tokenize based on tokenizer type
        if self._is_mlx_tokenizer():
            input_ids, attention_mask, labels = self._tokenize_with_mlx(input_text, conversation)
        else:
            input_ids, attention_mask, labels = self._tokenize_with_huggingface(input_text, conversation)

        # Convert to MLX arrays
        conversation.input_ids = mx.array(input_ids)
        conversation.attention_mask = mx.array(attention_mask)
        conversation.labels = mx.array(labels)

        return conversation

    def create_dataset(self, file_path: str | Path) -> list[ConversationExample]:
        """
        Create tokenized dataset from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of tokenized conversation examples
        """
        # Load raw conversations
        conversations = self.load_jsonl(file_path)

        # Tokenize all conversations
        tokenized_conversations = []
        logger.info("Tokenizing conversations...")

        for i, conversation in enumerate(conversations):
            try:
                tokenized_conv = self.tokenize_conversation(conversation)
                tokenized_conversations.append(tokenized_conv)

                if (i + 1) % 10 == 0:
                    logger.info("Tokenized %d/%d conversations", i + 1, len(conversations))

            except Exception as e:
                logger.warning("Failed to tokenize conversation %d: %s", i, e)
                continue

        logger.info("Successfully tokenized %d conversations", len(tokenized_conversations))
        return tokenized_conversations

    def create_batches(
        self, dataset: list[ConversationExample], batch_size: int = 4, shuffle: bool = True
    ) -> Iterator[dict[str, mx.array]]:
        """
        Create batches from tokenized dataset.

        Args:
            dataset: List of tokenized conversation examples
            batch_size: Size of each batch
            shuffle: Whether to shuffle the dataset

        Yields:
            Batches as dictionaries with MLX arrays
        """
        if shuffle:
            import random

            dataset = dataset.copy()
            random.shuffle(dataset)

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            # Stack examples into batch tensors
            input_ids_batch = mx.stack([conv.input_ids for conv in batch])
            attention_mask_batch = mx.stack([conv.attention_mask for conv in batch])
            labels_batch = mx.stack([conv.labels for conv in batch])

            yield {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
            }

    def get_data_stats(self, dataset: list[ConversationExample]) -> dict[str, Any]:
        """
        Get statistics about the dataset.

        Args:
            dataset: Tokenized dataset

        Returns:
            Dictionary with dataset statistics
        """
        if not dataset:
            return {}

        sequence_lengths = [len(conv.input_ids) for conv in dataset]
        prompt_lengths = []
        completion_lengths = []

        for conv in dataset:
            # Calculate prompt and completion lengths
            # Handle different tokenizer types (MLX vs HuggingFace)
            if hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "__call__"):
                # MLX TokenizerWrapper - use encode method
                prompt_tokens = self.tokenizer.encode(conv.prompt)
                completion_tokens = self.tokenizer.encode(conv.completion)
            else:
                # HuggingFace-style tokenizer
                prompt_tokens = self.tokenizer(conv.prompt, add_special_tokens=False)["input_ids"]
                completion_tokens = self.tokenizer(conv.completion, add_special_tokens=False)[
                    "input_ids"
                ]

            prompt_lengths.append(len(prompt_tokens))
            completion_lengths.append(len(completion_tokens))

        return {
            "num_examples": len(dataset),
            "max_sequence_length": max(sequence_lengths),
            "min_sequence_length": min(sequence_lengths),
            "avg_sequence_length": sum(sequence_lengths) / len(sequence_lengths),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_completion_length": sum(completion_lengths) / len(completion_lengths),
            "max_length_used": self.config.max_length,
        }


def create_data_loader(
    data_path: str | Path,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
) -> tuple[Iterator[dict[str, mx.array]], dict[str, Any]]:
    """
    Convenience function to create a data loader.

    Args:
        data_path: Path to JSONL data file
        tokenizer: Tokenizer for text processing
        batch_size: Batch size for training
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data

    Returns:
        Tuple of (batch_iterator, dataset_stats)
    """
    config = DatasetConfig(max_length=max_length)
    loader = ConversationDataLoader(tokenizer, config)

    # Create dataset
    dataset = loader.create_dataset(data_path)

    # Get statistics
    stats = loader.get_data_stats(dataset)

    # Create batch iterator
    batch_iterator = loader.create_batches(dataset, batch_size=batch_size, shuffle=shuffle)

    return batch_iterator, stats
