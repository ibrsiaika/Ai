"""
Training pipeline for India's GPT model.
Handles distributed training, mixed precision, checkpointing, and monitoring.
"""

import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    # Mixed precision
    fp16: bool = True
    fp16_opt_level: str = "O2"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Generation
    max_generate_length: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "india-gpt"
    seed: int = 42


class Trainer:
    """Main training class."""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader,
                 eval_dataloader,
                 config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=f"india-gpt-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config.__dict__
            )
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Total training steps: {len(train_dataloader) * config.num_train_epochs}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay."""
        # Get parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Warmup + Cosine annealing
        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
        
        return scheduler
    
    def train(self) -> Dict[str, Any]:
        """
        Training loop.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            epoch_loss = self._train_epoch(epoch)
            
            logger.info(f"Average train loss: {epoch_loss:.4f}")
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                logger.info(f"Evaluation loss: {eval_loss:.4f}")
                self.eval_losses.append(eval_loss)
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_model(os.path.join(self.config.output_dir, "best_model"))
                    logger.info(f"Saved best model with eval loss: {eval_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.config.output_dir, "final_model"))
        
        logger.info("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'best_eval_loss': self.best_eval_loss,
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Epoch number
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with autocast
            if self.config.fp16:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                self.global_step += 1
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
            
            # Evaluation and saving
            if (step + 1) % self.config.eval_steps == 0:
                if self.eval_dataloader is not None:
                    eval_loss = self._evaluate()
                    logger.info(f"Step {self.global_step}: Eval loss = {eval_loss:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({'eval_loss': eval_loss})
                    
                    self.model.train()
            
            if (step + 1) % self.config.save_steps == 0:
                save_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint-{self.global_step}"
                )
                self.save_model(save_path)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def _evaluate(self) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_model(self, save_path: str):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
        
        # Save scheduler state
        torch.save(self.scheduler.state_dict(), os.path.join(save_path, 'scheduler.pt'))
        
        # Save config
        config_dict = self.config.__dict__
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'model.pt'))
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'optimizer.pt'))
        )
        self.scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'scheduler.pt'))
        )
        
        logger.info(f"Model checkpoint loaded from {checkpoint_path}")


def create_trainer(model: nn.Module,
                  train_dataloader,
                  eval_dataloader,
                  config_dict: Dict) -> Trainer:
    """
    Create trainer with configuration dictionary.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        config_dict: Configuration dictionary
    
    Returns:
        Trainer instance
    """
    config = TrainingConfig(**config_dict)
    trainer = Trainer(model, train_dataloader, eval_dataloader, config)
    return trainer
