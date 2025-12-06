"""
Data loading and preprocessing pipeline for India's GPT model.
Handles NWorld dataset with support for multiple languages.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, DataLoader
import torch

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and prepare data for model training."""
    
    def __init__(self, 
                 tokenizer_name: str = "gpt2",
                 max_seq_length: int = 2048,
                 languages: List[str] = None):
        """
        Initialize data processor.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            max_seq_length: Maximum sequence length
            languages: List of languages to support (e.g., ['en', 'hi', 'ta'])
        """
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.languages = languages or ['en', 'hi', 'ta', 'te', 'ml', 'kn']
        
        # Load or create tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            logger.warning(f"Could not load tokenizer {tokenizer_name}, using default GPT2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_nworld_data(self, 
                        data_path: str,
                        split: str = 'train',
                        streaming: bool = False) -> Dataset:
        """
        Load NWorld dataset from local path or HuggingFace Hub.
        
        Args:
            data_path: Path to data directory or HuggingFace dataset name
            split: Dataset split ('train', 'validation', 'test')
            streaming: Whether to stream data (for large datasets)
        
        Returns:
            Loaded dataset
        """
        try:
            # Try loading from HuggingFace Hub first
            logger.info(f"Loading dataset from {data_path}")
            dataset = load_dataset(data_path, split=split, streaming=streaming)
        except:
            # Fall back to local loading
            logger.info(f"Loading dataset from local path {data_path}")
            if os.path.isfile(data_path):
                if data_path.endswith('.json'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    dataset = Dataset.from_dict({'text': data})
                elif data_path.endswith('.jsonl'):
                    texts = []
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            texts.append(json.loads(line).get('text', ''))
                    dataset = Dataset.from_dict({'text': texts})
            else:
                raise ValueError(f"Data path {data_path} not found")
        
        return dataset
    
    def preprocess_text(self, example: Dict) -> Dict:
        """
        Preprocess a single text example.
        
        Args:
            example: Dictionary with 'text' key
        
        Returns:
            Processed example with tokenized text
        """
        text = example.get('text', '')
        
        # Clean text
        text = text.strip()
        if not text:
            return {'input_ids': [], 'attention_mask': []}
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        
        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0],
            'token_type_ids': tokenized.get('token_type_ids', np.zeros(self.max_seq_length))[0]
        }
    
    def create_causal_lm_labels(self, batch: Dict) -> Dict:
        """
        Create labels for causal language modeling (GPT-style).
        Labels are shifted input_ids.
        
        Args:
            batch: Batch of examples
        
        Returns:
            Batch with labels added
        """
        batch['labels'] = batch['input_ids'].copy()
        batch['labels'] = np.where(
            batch['attention_mask'] == 1,
            batch['labels'],
            -100  # Ignore padding tokens in loss
        )
        return batch
    
    def prepare_dataset(self,
                       data_path: str,
                       split: str = 'train',
                       batch_size: int = 32,
                       num_workers: int = 4,
                       remove_columns: List[str] = None) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            data_path: Path to data
            split: Dataset split
            batch_size: Batch size
            num_workers: Number of workers
            remove_columns: Columns to remove after processing
        
        Returns:
            Processed dataset ready for training
        """
        # Load raw data
        dataset = self.load_nworld_data(data_path, split=split)
        
        # Preprocess
        dataset = dataset.map(
            self.preprocess_text,
            remove_columns=['text'],
            num_proc=num_workers,
            desc='Preprocessing texts'
        )
        
        # Create labels for causal LM
        dataset = dataset.map(
            self.create_causal_lm_labels,
            batched=True,
            desc='Creating labels'
        )
        
        # Set format for PyTorch
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return dataset


class LanguageMixingDataset(IterableDataset):
    """
    Iterable dataset that mixes multiple languages for training.
    Useful for multilingual model training.
    """
    
    def __init__(self, 
                 datasets_dict: Dict[str, Dataset],
                 sampling_strategy: str = 'uniform',
                 seed: int = 42):
        """
        Initialize language mixing dataset.
        
        Args:
            datasets_dict: Dictionary of {language: dataset}
            sampling_strategy: How to sample languages ('uniform', 'proportional', 'temperature')
            seed: Random seed
        """
        self.datasets = datasets_dict
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate dataset sizes
        self.sizes = {lang: len(ds) for lang, ds in datasets_dict.items()}
        self.total_size = sum(self.sizes.values())
        
        # Calculate sampling probabilities
        if sampling_strategy == 'uniform':
            self.probs = {lang: 1.0 / len(datasets_dict) for lang in datasets_dict}
        elif sampling_strategy == 'proportional':
            self.probs = {lang: size / self.total_size for lang, size in self.sizes.items()}
        elif sampling_strategy == 'temperature':
            # Temperature-based sampling
            temp = 0.7
            sizes_temp = {lang: size ** (1.0 / temp) for lang, size in self.sizes.items()}
            total_temp = sum(sizes_temp.values())
            self.probs = {lang: size_t / total_temp for lang, size_t in sizes_temp.items()}
        
        logger.info(f"Language probabilities: {self.probs}")
    
    def __iter__(self):
        """Iterate through mixed language data."""
        iterators = {lang: iter(ds) for lang, ds in self.datasets.items()}
        languages = list(self.datasets.keys())
        
        while True:
            # Sample a language
            lang = self.rng.choice(languages, p=list(self.probs.values()))
            
            try:
                yield next(iterators[lang])
            except StopIteration:
                # Reset iterator for this language
                iterators[lang] = iter(self.datasets[lang])
                try:
                    yield next(iterators[lang])
                except StopIteration:
                    break


class DataCollatorForLanguageModeling:
    """Custom data collator for language modeling tasks."""
    
    def __init__(self, 
                 tokenizer,
                 mlm: bool = False,
                 mlm_probability: float = 0.15):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer to use
            mlm: Whether to use masked language modeling (for BERT-style models)
            mlm_probability: Probability of masking a token
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of examples.
        
        Args:
            batch: List of examples
        
        Returns:
            Collated batch
        """
        # Stack tensors
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if self.mlm:
            # Apply masking
            input_ids_copy = input_ids.clone()
            mask_indices = torch.rand(input_ids.shape) < self.mlm_probability
            
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    if mask_indices[i, j]:
                        input_ids_copy[i, j] = self.tokenizer.mask_token_id
            
            result['input_ids'] = input_ids_copy
        
        return result


def create_data_loaders(data_config: Dict,
                       tokenizer_name: str = 'gpt2') -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_config: Configuration dictionary with data paths and settings
        tokenizer_name: Name of tokenizer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    processor = DataProcessor(tokenizer_name=tokenizer_name,
                             max_seq_length=data_config.get('max_seq_length', 2048))
    
    # Load datasets
    train_dataset = processor.prepare_dataset(
        data_config['train_path'],
        split='train',
        num_workers=data_config.get('num_workers', 4)
    )
    
    val_dataset = processor.prepare_dataset(
        data_config.get('val_path', data_config['train_path']),
        split='validation',
        num_workers=data_config.get('num_workers', 4)
    )
    
    # Create collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=data_config.get('mlm', False)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Set to 0 for PyTorch Dataset
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    return train_loader, val_loader
