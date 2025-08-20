"""
MLX-optimized data loading for LoRA fine-tuning.

Handles JSONL conversation data, tokenization, and batch creation
with efficient MLX tensor operations and Apple Silicon optimization.
"""

import json
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from transformers import AutoTokenizer
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    add_special_tokens: bool = True
    chat_template: Optional[str] = None
    ignore_index: int = -100


@dataclass
class ConversationExample:
    """Container for a single conversation example."""
    
    prompt: str
    completion: str
    input_ids: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    labels: Optional[mx.array] = None


class ConversationDataLoader:
    """
    MLX-optimized data loader for conversation data.
    
    Loads JSONL files with conversation data and creates MLX-compatible
    batches for training with efficient memory usage and Apple Silicon optimization.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        config: Optional[DatasetConfig] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            tokenizer: Tokenizer for text processing
            config: Dataset configuration
        """
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for tokenization")
    
    def load_jsonl(self, file_path: Union[str, Path]) -> List[ConversationExample]:
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
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract prompt and completion
                    if 'prompt' in data and 'completion' in data:
                        conversation = ConversationExample(
                            prompt=data['prompt'],
                            completion=data['completion']
                        )
                        conversations.append(conversation)
                    else:
                        print(f"Warning: Skipping line {line_num}, missing 'prompt' or 'completion'")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(conversations)} conversations from {file_path}")
        return conversations
    
    def tokenize_conversation(self, conversation: ConversationExample) -> ConversationExample:
        """
        Tokenize a single conversation example.
        
        Args:
            conversation: Raw conversation example
            
        Returns:
            Tokenized conversation example
        """
        # Create input text by combining prompt and completion
        if self.config.chat_template:
            # Use custom chat template if provided
            input_text = self.config.chat_template.format(
                prompt=conversation.prompt, 
                completion=conversation.completion
            )
        else:
            # Default format
            input_text = f"{conversation.prompt} {conversation.completion}"
        
        # Handle different tokenizer types (MLX vs HuggingFace)
        if hasattr(self.tokenizer, 'encode') and not hasattr(self.tokenizer, '__call__'):
            # MLX TokenizerWrapper - use encode method
            input_ids = self.tokenizer.encode(input_text)
            prompt_ids = self.tokenizer.encode(conversation.prompt)
            
            # Truncate if necessary
            if len(input_ids) > self.config.max_length:
                input_ids = input_ids[:self.config.max_length]
            
            # Pad if necessary
            if self.config.padding and len(input_ids) < self.config.max_length:
                pad_length = self.config.max_length - len(input_ids)
                # Assuming 0 is the pad token - this might need adjustment
                input_ids = input_ids + [0] * pad_length
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * min(len(input_ids), self.config.max_length)
            if len(attention_mask) < self.config.max_length:
                attention_mask += [0] * (self.config.max_length - len(attention_mask))
            
            # Create labels - mask prompt tokens with ignore_index
            labels = input_ids.copy()
            prompt_length = len(prompt_ids)
            if prompt_length < len(labels):
                labels[:prompt_length] = [self.config.ignore_index] * prompt_length
            
        else:
            # HuggingFace-style tokenizer
            encoding = self.tokenizer(
                input_text,
                max_length=self.config.max_length,
                padding='max_length' if self.config.padding else False,
                truncation=self.config.truncation,
                add_special_tokens=self.config.add_special_tokens,
                return_tensors="np"
            )
            
            # Create labels for causal language modeling
            prompt_encoding = self.tokenizer(
                conversation.prompt,
                add_special_tokens=self.config.add_special_tokens,
                return_tensors="np"
            )
            prompt_length = len(prompt_encoding['input_ids'][0])
            
            # Create labels - mask prompt tokens with ignore_index
            labels = encoding['input_ids'].copy()
            labels[0, :prompt_length] = self.config.ignore_index
            
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            labels = labels[0]
        
        # Convert to MLX arrays
        conversation.input_ids = mx.array(input_ids)
        conversation.attention_mask = mx.array(attention_mask)
        conversation.labels = mx.array(labels)
        
        return conversation
    
    def create_dataset(self, file_path: Union[str, Path]) -> List[ConversationExample]:
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
        print("Tokenizing conversations...")
        
        for i, conversation in enumerate(conversations):
            try:
                tokenized_conv = self.tokenize_conversation(conversation)
                tokenized_conversations.append(tokenized_conv)
                
                if (i + 1) % 10 == 0:
                    print(f"Tokenized {i + 1}/{len(conversations)} conversations")
                    
            except Exception as e:
                print(f"Warning: Failed to tokenize conversation {i}: {e}")
                continue
        
        print(f"Successfully tokenized {len(tokenized_conversations)} conversations")
        return tokenized_conversations
    
    def create_batches(
        self, 
        dataset: List[ConversationExample], 
        batch_size: int = 4, 
        shuffle: bool = True
    ) -> Iterator[Dict[str, mx.array]]:
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
            batch = dataset[i:i + batch_size]
            
            # Stack examples into batch tensors
            input_ids_batch = mx.stack([conv.input_ids for conv in batch])
            attention_mask_batch = mx.stack([conv.attention_mask for conv in batch])
            labels_batch = mx.stack([conv.labels for conv in batch])
            
            yield {
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'labels': labels_batch
            }
    
    def get_data_stats(self, dataset: List[ConversationExample]) -> Dict[str, Any]:
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
            if hasattr(self.tokenizer, 'encode') and not hasattr(self.tokenizer, '__call__'):
                # MLX TokenizerWrapper - use encode method
                prompt_tokens = self.tokenizer.encode(conv.prompt)
                completion_tokens = self.tokenizer.encode(conv.completion)
            else:
                # HuggingFace-style tokenizer
                prompt_tokens = self.tokenizer(conv.prompt, add_special_tokens=False)['input_ids']
                completion_tokens = self.tokenizer(conv.completion, add_special_tokens=False)['input_ids']
            
            prompt_lengths.append(len(prompt_tokens))
            completion_lengths.append(len(completion_tokens))
        
        return {
            'num_examples': len(dataset),
            'max_sequence_length': max(sequence_lengths),
            'min_sequence_length': min(sequence_lengths),
            'avg_sequence_length': sum(sequence_lengths) / len(sequence_lengths),
            'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths),
            'avg_completion_length': sum(completion_lengths) / len(completion_lengths),
            'max_length_used': self.config.max_length,
        }


def create_data_loader(
    data_path: Union[str, Path],
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True
) -> Tuple[Iterator[Dict[str, mx.array]], Dict[str, Any]]:
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