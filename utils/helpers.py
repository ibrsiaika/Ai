"""
Utility functions for India's GPT model.
Includes helpers for model management, data processing, and deployment.
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model saving, loading, and versioning."""
    
    def __init__(self, model_dir: str = './models'):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self,
                  model: torch.nn.Module,
                  name: str,
                  metadata: Optional[Dict] = None) -> str:
        """
        Save model with metadata.
        
        Args:
            model: Model to save
            name: Model name
            metadata: Additional metadata
        
        Returns:
            Path to saved model
        """
        save_dir = self.model_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = save_dir / 'model.pt'
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['saved_at'] = datetime.now().isoformat()
        metadata_path = save_dir / 'metadata.json'
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
        return str(save_dir)
    
    def load_model(self,
                  model_class,
                  name: str,
                  device: str = 'cuda') -> torch.nn.Module:
        """
        Load model from saved checkpoint.
        
        Args:
            model_class: Model class to instantiate
            name: Model name
            device: Device to load on
        
        Returns:
            Loaded model
        """
        model_dir = self.model_dir / name
        model_path = model_dir / 'model.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        logger.info(f"Model loaded from {model_dir}")
        return model
    
    def list_models(self) -> List[str]:
        """List all available models."""
        models = [d.name for d in self.model_dir.iterdir() if d.is_dir()]
        return sorted(models)


class ConfigManager:
    """Manage configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    @staticmethod
    def save_config(config: Dict, save_path: str):
        """Save configuration to file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @staticmethod
    def merge_configs(base_config: Dict,
                     override_config: Dict) -> Dict:
        """Merge two configurations."""
        merged = base_config.copy()
        merged.update(override_config)
        return merged


class DataUtils:
    """Utility functions for data processing."""
    
    @staticmethod
    def split_text(text: str,
                  max_length: int = 512,
                  overlap: int = 128) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            max_length: Maximum chunk length
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_length, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text for model training.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (optional)
        # text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\"-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def create_data_splits(data: List[str],
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: List of data samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
        
        Returns:
            Tuple of (train, val, test)
        """
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data


class TextProcessor:
    """Text processing utilities."""
    
    @staticmethod
    def tokenize_hindi(text: str) -> List[str]:
        """
        Tokenize Hindi text.
        
        Args:
            text: Hindi text
        
        Returns:
            List of tokens
        """
        # Simple word boundary tokenization
        tokens = text.split()
        return tokens
    
    @staticmethod
    def tokenize_english(text: str) -> List[str]:
        """
        Tokenize English text.
        
        Args:
            text: English text
        
        Returns:
            List of tokens
        """
        import re
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    @staticmethod
    def tokenize_multilingual(text: str,
                            language: str = 'en') -> List[str]:
        """
        Tokenize multilingual text.
        
        Args:
            text: Text to tokenize
            language: Language code
        
        Returns:
            List of tokens
        """
        if language == 'hi':
            return TextProcessor.tokenize_hindi(text)
        elif language == 'en':
            return TextProcessor.tokenize_english(text)
        else:
            return text.split()


class PerformanceMonitor:
    """Monitor training and inference performance."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
    
    def log_metric(self,
                  name: str,
                  value: float,
                  step: Optional[int] = None):
        """
        Log a metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step or len(self.metrics[name])
        })
    
    def get_metric(self, name: str, step: Optional[int] = None):
        """
        Get metric value.
        
        Args:
            name: Metric name
            step: Training step
        
        Returns:
            Metric value(s)
        """
        if name not in self.metrics:
            return None
        
        if step is None:
            return [m['value'] for m in self.metrics[name]]
        else:
            for m in self.metrics[name]:
                if m['step'] == step:
                    return m['value']
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            metric_values = [m['value'] for m in values]
            summary[name] = {
                'min': min(metric_values),
                'max': max(metric_values),
                'mean': sum(metric_values) / len(metric_values),
                'last': metric_values[-1]
            }
        return summary


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
