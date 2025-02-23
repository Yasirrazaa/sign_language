"""Training implementation for sign language detection models."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
<<<<<<< HEAD
# Remove CosineAnnealingWarmRestarts import
=======
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
>>>>>>> 3ece852 (Add initial project structure and essential files for sign language detection)
from torch.utils.data import DataLoader
import wandb
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
from dataclasses import dataclass

from ..models import SignLanguageCNNLSTM, VideoTransformer
from ..utils import get_checkpoint_dir
<<<<<<< HEAD
from .callbacks import ModelCheckpoint, EarlyStopping, WarmupScheduler
=======
from .callbacks import ModelCheckpoint, EarlyStopping
>>>>>>> 3ece852 (Add initial project structure and essential files for sign language detection)
from .metrics import calculate_metrics
from ..config import TRAIN_CONFIG

@dataclass
class TrainerConfig:
    """Configuration for training."""
    num_epochs: int = TRAIN_CONFIG['epochs']
    learning_rate: float = TRAIN_CONFIG['initial_learning_rate']
    weight_decay: float = TRAIN_CONFIG['weight_decay']
    clip_grad_norm: float = TRAIN_CONFIG['clip_grad_norm']
    label_smoothing: float = TRAIN_CONFIG['label_smoothing']
    warmup_epochs: int = TRAIN_CONFIG['warmup_epochs']
    use_wandb: bool = True
    checkpoint_dir: Optional[Path] = None
    device: Optional[torch.device] = None
    fold: Optional[int] = None  # Current fold number for cross-validation

class Trainer:
    """Model trainer implementation."""
    
    def __init__(
        self,
        model: Union[SignLanguageCNNLSTM, VideoTransformer],
        config: TrainerConfig
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.start_time = time.time()
        
        # Set device
        self.device = (
            config.device or
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
<<<<<<< HEAD
        # Initialize scheduler with warmup
        self.scheduler = WarmupScheduler(
            optimizer=self.optimizer,
            warmup_epochs=config.warmup_epochs,
            initial_lr=config.learning_rate,
            min_lr=TRAIN_CONFIG['min_learning_rate']
=======
        # Initialize scheduler with cosine annealing and warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart interval
            T_mult=2,  # Multiply interval by 2 after each restart
            eta_min=TRAIN_CONFIG['min_learning_rate']
>>>>>>> 3ece852 (Add initial project structure and essential files for sign language detection)
        )
        
        # Initialize loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup checkpoint directory
        self.checkpoint_dir = (
            config.checkpoint_dir or
            get_checkpoint_dir()
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize history tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_top3_acc': [], 'val_top3_acc': [],
            'train_top5_acc': [], 'val_top5_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': []
        }
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._setup_wandb()
    
    def _get_warmup_lr(self, epoch: int, batch_idx: int, num_batches: int) -> float:
        """Calculate learning rate during warmup period."""
        if epoch >= self.config.warmup_epochs:
            return None
            
        total_steps = self.config.warmup_epochs * num_batches
        current_step = epoch * num_batches + batch_idx
        
        return self.config.learning_rate * (current_step / total_steps)
    
    def _setup_callbacks(self) -> List[Callable]:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                monitor='val_loss',
                mode='min',
                save_top_k=3
            )
        )
        
        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=TRAIN_CONFIG['early_stopping_patience'],
                mode='min'
            )
        )
        
        return callbacks
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project="sign-language-detection",
            config={
                "model_type": self.model.__class__.__name__,
                **self.config.__dict__
            }
        )
        wandb.watch(self.model)
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        epoch: int,
        batch_idx: int,
        num_batches: int
    ) -> Dict[str, float]:
        """Single training step."""
        frames, labels = batch
        frames = frames.to(self.device)
        # Convert one-hot labels to class indices
        labels = torch.argmax(labels, dim=1).to(self.device)
        
        # Warmup learning rate if in warmup period
        warmup_lr = self._get_warmup_lr(epoch, batch_idx, num_batches)
        if warmup_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Forward pass
        predictions = self.model(frames)
        
        # Calculate loss
        loss = self.criterion(predictions, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm
            )
        
        self.optimizer.step()
        
        # Step scheduler if not in warmup
        if warmup_lr is None:
            self.scheduler.step(epoch + batch_idx / num_batches)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions.detach(), labels.detach())
        metrics['loss'] = loss.item()
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        # Prefix metrics with 'train_'
        train_metrics = {f'train_{k}': v for k, v in metrics.items()}
        
        return train_metrics
    
    @torch.no_grad()
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single validation step."""
        frames, labels = batch
        frames = frames.to(self.device)
        # Convert one-hot labels to class indices
        labels = torch.argmax(labels, dim=1).to(self.device)
        
        # Forward pass
        predictions = self.model(frames)
        
        # Calculate loss
        loss = self.criterion(predictions, labels)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels)
        metrics['loss'] = loss.item()
        
        # Prefix metrics with 'val_'
        val_metrics = {f'val_{k}': v for k, v in metrics.items()}
        
        return val_metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
            metrics = self.train_step(batch, epoch, batch_idx, num_batches)
            epoch_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = []
        
        if len(val_loader.dataset) == 0:
            self.logger.warning("No validation data available!")
            return {
                'val_loss': 0.0,
                'val_accuracy': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0
            }
        
        for batch in tqdm(val_loader, desc='Validating'):
            metrics = self.validate_step(batch)
            epoch_metrics.append(metrics)
        
        if not epoch_metrics:
            self.logger.warning("No validation batches completed!")
            return {
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0
            }
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            for key, value in train_metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            for key, value in val_metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Track learning rate
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Run callbacks
            stop_training = False
            for callback in self.callbacks:
                if callback(self, val_metrics):
                    stop_training = True
                    break
            
            if stop_training:
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Training completed")
        return self.history
    
    def _format_time(self, seconds: float) -> str:
        """Format time in hours:minutes:seconds."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f'{h:d}:{m:02d}:{s:02d}'

    def _format_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        time_elapsed: float,
        time_remaining: float
    ) -> str:
        """Format metrics for display."""
        fold_str = f"Fold {self.config.fold}/{TRAIN_CONFIG['num_folds']} | " if self.config.fold else ""
        
        metrics_str = (
            f"\n{'='*100}\n"
            f"{fold_str}Epoch [{epoch+1}/{self.config.num_epochs}] | "
            f"Time {self._format_time(time_elapsed)} (ETA: {self._format_time(time_remaining)})\n"
            f"{'='*100}\n"
            f"Training   | Loss: {train_metrics['train_loss']:.4f} | "
            f"Acc: {train_metrics['train_accuracy']:.2%} | "
            f"Top-3: {train_metrics['train_top3_acc']:.2%} | "
            f"F1: {train_metrics['train_f1']:.4f}\n"
            f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_accuracy']:.2%} | "
            f"Top-3: {val_metrics['val_top3_acc']:.2%} | "
            f"F1: {val_metrics['val_f1']:.4f}\n"
            f"{'='*100}\n"
            f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
            f"{'='*100}"
        )
        
        return metrics_str

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics for current epoch."""
        # Calculate timing information
        time_elapsed = time.time() - self.start_time
        time_per_epoch = time_elapsed / (epoch + 1)
        time_remaining = time_per_epoch * (self.config.num_epochs - epoch - 1)
        
        # Format and display metrics
        metrics_str = self._format_metrics(
            epoch, train_metrics, val_metrics,
            time_elapsed, time_remaining
        )
        self.logger.info(metrics_str)
        
        # W&B logging
        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
