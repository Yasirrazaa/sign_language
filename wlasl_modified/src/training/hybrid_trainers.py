"""Specialized trainers for hybrid transformer architectures."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
import wandb
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import gc

from ..models.hybrid_transformers import CNNTransformer, TimeSformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseTransformerTrainer:
    """Base trainer for transformer architectures."""
    
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
        Initialize base trainer.
        
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
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mixed precision training
        self.scaler = GradScaler(enabled=config['mixed_precision'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Memory tracking
        self.memory_stats = []
    
    def _check_memory(self) -> Dict[str, float]:
        """Check current memory usage."""
        stats = {
            'ram_used': torch.cuda.memory_allocated() / 1024**3,
            'ram_reserved': torch.cuda.memory_reserved() / 1024**3
        }
        self.memory_stats.append(stats)
        return stats
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

class CNNTransformerTrainer(BaseTransformerTrainer):
    """Specialized trainer for CNN-Transformer architecture."""
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = {'loss': 0.0, 'accuracy': 0.0}
        num_batches = len(self.train_loader)
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        accumulated_loss = 0
        
        with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch + 1}') as pbar:
            for batch_idx, (frames, labels) in enumerate(self.train_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision forward pass
                with autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    # Gradient clipping
                    if self.config['gradient_clip'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    metrics['loss'] += accumulated_loss
                    accumulated_loss = 0
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == labels).float().mean().item()
                    metrics['accuracy'] += acc
                
                # Memory management
                if batch_idx % self.config['cleanup_interval'] == 0:
                    self._cleanup_memory()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': metrics['loss'] / (batch_idx + 1),
                    'acc': metrics['accuracy'] / (batch_idx + 1)
                })
        
        # Average metrics
        metrics['loss'] /= (num_batches // self.config['gradient_accumulation_steps'])
        metrics['accuracy'] /= (num_batches // self.config['gradient_accumulation_steps'])
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        metrics = {'val_loss': 0.0, 'val_accuracy': 0.0}
        num_batches = len(self.val_loader)
        
        for frames, labels in tqdm(self.val_loader, desc='Validation'):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision inference
            with autocast(enabled=self.config['mixed_precision']):
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
            
            # Update metrics
            metrics['val_loss'] += loss.item()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item()
            metrics['val_accuracy'] += acc
        
        # Average metrics
        metrics['val_loss'] /= num_batches
        metrics['val_accuracy'] /= num_batches
        
        return metrics

class TimeSformerTrainer(BaseTransformerTrainer):
    """Specialized trainer for TimeSformer architecture."""
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with space-time attention."""
        self.model.train()
        metrics = {'loss': 0.0, 'accuracy': 0.0}
        num_batches = len(self.train_loader)
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        accumulated_loss = 0
        
        with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch + 1}') as pbar:
            for batch_idx, (frames, labels) in enumerate(self.train_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision forward pass with chunked processing
                with autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    # Gradient clipping
                    if self.config['gradient_clip'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['gradient_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    metrics['loss'] += accumulated_loss
                    accumulated_loss = 0
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == labels).float().mean().item()
                    metrics['accuracy'] += acc
                
                # Memory management with special attention to transformer cache
                if batch_idx % self.config['cleanup_interval'] == 0:
                    self._cleanup_memory()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': metrics['loss'] / (batch_idx + 1),
                    'acc': metrics['accuracy'] / (batch_idx + 1)
                })
        
        # Average metrics
        metrics['loss'] /= (num_batches // self.config['gradient_accumulation_steps'])
        metrics['accuracy'] /= (num_batches // self.config['gradient_accumulation_steps'])
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model with chunked processing."""
        self.model.eval()
        metrics = {'val_loss': 0.0, 'val_accuracy': 0.0}
        num_batches = len(self.val_loader)
        
        for frames, labels in tqdm(self.val_loader, desc='Validation'):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision inference with chunked processing
            with autocast(enabled=self.config['mixed_precision']):
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
            
            # Update metrics
            metrics['val_loss'] += loss.item()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item()
            metrics['val_accuracy'] += acc
            
            # Clear cache after each batch
            if self.config['aggressive_cleanup']:
                self._cleanup_memory()
        
        # Average metrics
        metrics['val_loss'] /= num_batches
        metrics['val_accuracy'] /= num_batches
        
        return metrics

def create_trainer(
    model_name: str,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict,
    checkpoint_dir: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> BaseTransformerTrainer:
    """
    Create appropriate trainer based on model type.
    
    Args:
        model_name: Name of the model ('cnn_transformer' or 'timesformer')
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Training configuration
        checkpoint_dir: Directory for checkpoints
        scheduler: Optional learning rate scheduler
        
    Returns:
        Initialized trainer
    """
    if model_name == 'cnn_transformer':
        return CNNTransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler
        )
    elif model_name == 'timesformer':
        return TimeSformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")