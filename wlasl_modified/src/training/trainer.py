"""Memory-efficient training implementation for sign language recognition."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional
import logging
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import gc
import psutil
import wandb
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientTrainer:
    """Memory-efficient model trainer with gradient accumulation and mixed precision."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 config: Dict,
                 checkpoint_dir: Path,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize trainer with memory efficiency features.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
            checkpoint_dir: Directory for checkpoints
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mixed precision training
        self.scaler = GradScaler(enabled=config['use_mixed_precision'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        
        # Memory monitoring
        self.memory_stats = []
    
    def _check_memory_usage(self):
        """Monitor memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'rss': memory_info.rss / 1024**2,  # RSS in MB
            'vms': memory_info.vms / 1024**2,  # VMS in MB
            'gpu_memory': torch.cuda.memory_allocated() / 1024**2  # GPU memory in MB
        }
        
        self.memory_stats.append(stats)
        
        # Log memory usage
        if wandb.run is not None:
            wandb.log({"memory/{}".format(k): v for k, v in stats.items()})
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with memory efficiency.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        # Initialize loss accumulator
        accumulated_loss = 0
        optimizer_steps = 0
        
        with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch + 1}') as pbar:
            for batch_idx, (frames, labels) in enumerate(self.train_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision training
                with autocast(enabled=self.config['use_mixed_precision']):
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Optimizer step every accumulation_steps batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config['gradient_clip'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    metrics['loss'] += accumulated_loss
                    accumulated_loss = 0
                    optimizer_steps += 1
                    
                    # Memory cleanup
                    if self.config['aggressive_memory_cleanup']:
                        del frames, labels, outputs, loss
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': metrics['loss'] / (optimizer_steps + 1e-8)})
                
                # Monitor memory usage
                if batch_idx % 100 == 0:
                    self._check_memory_usage()
        
        # Average metrics
        metrics['loss'] /= optimizer_steps
        
        return dict(metrics)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model with memory efficiency.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        for frames, labels in tqdm(self.val_loader, desc='Validation'):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision inference
            with autocast(enabled=self.config['use_mixed_precision']):
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
            
            # Update metrics
            metrics['loss'] += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            metrics['accuracy'] += (preds == labels).float().mean().item()
            
            # Memory cleanup
            if self.config['aggressive_memory_cleanup']:
                del frames, labels, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return dict(metrics)
    
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_model_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def train(self, num_epochs: int) -> Dict:
        """
        Train model with memory efficiency.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary of training history
        """
        try:
            history = defaultdict(list)
            
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch()
                
                # Validation phase
                val_metrics = self.validate()
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Update metrics
                metrics = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                }
                
                # Log metrics
                if wandb.run is not None:
                    wandb.log(metrics)
                
                # Update history
                for k, v in metrics.items():
                    history[k].append(v)
                
                # Save checkpoint
                self.save_checkpoint({
                    **metrics,
                    'val_loss': val_metrics['loss']
                })
                
                # Early stopping
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break
                
                # Log epoch summary
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"train_loss: {train_metrics['loss']:.4f} - "
                    f"val_loss: {val_metrics['loss']:.4f} - "
                    f"val_accuracy: {val_metrics['accuracy']:.4f}"
                )
                
                # Memory cleanup between epochs
                gc.collect()
                torch.cuda.empty_cache()
            
            # Save training history
            history_path = self.checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            return dict(history)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return dict(history)
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        finally:
            # Save memory statistics
            memory_stats_path = self.checkpoint_dir / 'memory_stats.json'
            with open(memory_stats_path, 'w') as f:
                json.dump(self.memory_stats, f, indent=2)

if __name__ == '__main__':
    # Example configuration
    config = {
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'gradient_clip': 1.0,
        'early_stopping_patience': 10,
        'aggressive_memory_cleanup': True
    }
    
    # Example usage
    model = nn.Module()
    trainer = MemoryEfficientTrainer(
        model=model,
        train_loader=None,  # Add your data loader
        val_loader=None,    # Add your data loader
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device('cuda'),
        config=config,
        checkpoint_dir='checkpoints'
    )
    
    history = trainer.train(num_epochs=100)
