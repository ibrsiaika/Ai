"""
Main training script for India's GPT model.
Comprehensive end-to-end training pipeline.
"""

import os
import sys
import logging
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Optional

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.data_loader import DataProcessor, create_data_loaders
from models.gpt_model import create_gpt_model, GPTConfig
from training.trainer import Trainer, TrainingConfig
from inference.generator import TextGenerator, GenerationConfig
from utils.metrics import create_metrics_calculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_arg_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Train India's GPT model"
    )
    
    # Data arguments
    parser.add_argument(
        '--train_data',
        type=str,
        default='wikitext',
        help='Training data path or HuggingFace dataset name'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Validation data path'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=2048,
        help='Maximum sequence length'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_size',
        type=str,
        default='base',
        choices=['small', 'base', 'large', 'xlarge'],
        help='Model size'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,
        help='Vocabulary size'
    )
    
    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=1000,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluation frequency'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=1000,
        help='Checkpoint saving frequency'
    )
    
    # Device and optimization
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Monitoring
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for monitoring'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='india-gpt',
        help='Weights & Biases project name'
    )
    
    # Generation
    parser.add_argument(
        '--generate_samples',
        action='store_true',
        help='Generate samples after training'
    )
    parser.add_argument(
        '--sample_prompts',
        type=str,
        default='test,hello,India,technology',
        help='Comma-separated prompts for generation'
    )
    
    return parser


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict, output_dir: str):
    """Save configuration to file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


def main():
    """Main training function."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    logger.info("=" * 50)
    logger.info("India's GPT Model - Training Pipeline")
    logger.info("=" * 50)
    
    # Print arguments
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    save_config(config_dict, args.output_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Step 1: Load data
    logger.info("\n" + "=" * 50)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 50)
    
    data_config = {
        'train_path': args.train_data,
        'val_path': args.val_data or args.train_data,
        'max_seq_length': args.max_seq_length,
        'batch_size': args.batch_size,
        'num_workers': 4
    }
    
    try:
        train_loader, val_loader = create_data_loaders(
            data_config,
            tokenizer_name='gpt2'
        )
        logger.info(f"Loaded training data: {len(train_loader)} batches")
        logger.info(f"Loaded validation data: {len(val_loader)} batches")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Proceeding with empty data loaders for code testing")
        train_loader = []
        val_loader = []
    
    # Step 2: Create model
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Creating Model")
    logger.info("=" * 50)
    
    model = create_gpt_model(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.max_seq_length
    )
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Step 3: Setup training
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Setting Up Training")
    logger.info("=" * 50)
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        config=training_config
    )
    
    # Step 4: Train
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Training Model")
    logger.info("=" * 50)
    
    if train_loader:
        results = trainer.train()
        logger.info(f"Training Results: {results}")
    else:
        logger.info("Skipping training due to empty data loaders")
        trainer.save_model(os.path.join(args.output_dir, 'untrained_model'))
    
    # Step 5: Generation (Optional)
    if args.generate_samples:
        logger.info("\n" + "=" * 50)
        logger.info("Step 5: Generating Samples")
        logger.info("=" * 50)
        
        from transformers import AutoTokenizer
        
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            generator = TextGenerator(model, tokenizer, device=device)
            
            prompts = [p.strip() for p in args.sample_prompts.split(',')]
            
            for prompt in prompts:
                logger.info(f"\nPrompt: {prompt}")
                
                config = GenerationConfig(
                    max_length=128,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95
                )
                
                generated = generator.generate(prompt, config)
                logger.info(f"Generated: {generated}")
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Training Pipeline Completed!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
