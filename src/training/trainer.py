"""Training implementation for sign language detection."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
import logging
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
from dataclasses import dataclass

from ..utils import get_checkpoint_dir
from .callbacks import ModelCheckpoint, EarlyStopping
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
    mixed_precision: bool = True  # Enable mixed precision training
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    max_grad_norm: float = 1.0  # Maximum gradient norm
    enable_checkpointing: bool = True  # Enable gradient checkpointing

class MemoryEfficientTrainer:
    """GPU memory-efficient model trainer implementation."""
    
    def __init__(
        self,
        model: nn.Module,
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
        
        # Enable gradient checkpointing if supported
        if config.enable_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Initialize optimizer with weight decay for normalization layers
        self._setup_optimizer()
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
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
        self.history = self._init_history()
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay for normalization layers."""
        # Separate parameters that should and shouldn't use weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )
        
        # Linear warmup and cosine decay
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=1,  # Will be updated in train()
            pct_start=self.config.warmup_epochs / self.config.num_epochs,
            anneal_strategy='cos'
        )

    def _init_history(self) -> Dict[str, List]:
        """Initialize training history."""
        return {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_top3_acc': [], 'val_top3_acc': [],
            'train_top5_acc': [], 'val_top5_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': [],
            'gpu_memory': []
        }

    def _setup_callbacks(self) -> List[Callable]:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                monitor='val_loss',
                mode='min',
                save_top_k=2  # Reduced from 3 to save memory
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
    
    def _clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """Single training step with memory optimizations."""
        frames, labels = batch
        frames = frames.to(self.device)
        labels = torch.argmax(labels, dim=1).to(self.device)
        
        # Mixed precision training
        if self.scaler is not None:
            with autocast():
                predictions = self.model(frames)
                loss = self.criterion(predictions, labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            predictions = self.model(frames)
            loss = self.criterion(predictions, labels)
            loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Calculate metrics
        metrics = calculate_metrics(predictions.detach(), labels.detach())
        metrics['loss'] = loss.item() * self.config.gradient_accumulation_steps
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        if torch.cuda.is_available():
            metrics['gpu_memory'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # Clear unnecessary tensors
        del frames, labels, predictions
        
        return {f'train_{k}': v for k, v in metrics.items()}

    @torch.no_grad()
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single validation step."""
        frames, labels = batch
        frames = frames.to(self.device)
        labels = torch.argmax(labels, dim=1).to(self.device)
        
        if self.scaler is not None:
            with autocast():
                predictions = self.model(frames)
                loss = self.criterion(predictions, labels)
        else:
            predictions = self.model(frames)
            loss = self.criterion(predictions, labels)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, labels)
        metrics['loss'] = loss.item()
        
        # Clear memory
        del frames, labels, predictions
        
        return {f'val_{k}': v for k, v in metrics.items()}

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with memory optimizations."""
        self.model.train()
        epoch_metrics = []
        
        for step, batch in enumerate(tqdm(train_loader, desc='Training')):
            metrics = self.train_step(batch, step)
            epoch_metrics.append(metrics)
            
            # Step scheduler
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scheduler.step()
            
            # Clear memory periodically
            if step % 10 == 0:
                self._clear_memory()
        
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
        """Train the model with memory optimizations."""
        self.logger.info("Starting training...")
        
        # Update scheduler steps
        self.scheduler.total_steps = len(train_loader) * self.config.num_epochs
        
        try:
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
                
                # Clear memory before callbacks
                self._clear_memory()
                
                # Run callbacks
                stop_training = False
                for callback in self.callbacks:
                    if callback(self, val_metrics):
                        stop_training = True
                        break
                
                if stop_training:
                    self.logger.info("Early stopping triggered")
                    break
                
        except Exception as e:
            self.logger.error(f"Training interrupted: {str(e)}")
            raise
        finally:
            # Clean up
            self._clear_memory()
            if self.config.use_wandb:
                wandb.finish()
        
        self.logger.info("Training completed")
        return self.history

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
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0
            }
        
        for batch in tqdm(val_loader, desc='Validating'):
            metrics = self.validate_step(batch)
            epoch_metrics.append(metrics)
            
            # Clear memory periodically
            if len(epoch_metrics) % 10 == 0:
                self._clear_memory()
        
        # Average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics

    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint efficiently."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler']:
            self.scaler.load_state_dict(checkpoint['scaler'])

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics."""
        # Calculate timing information
        time_elapsed = time.time() - self.start_time
        time_per_epoch = time_elapsed / (epoch + 1)
        time_remaining = time_per_epoch * (self.config.num_epochs - epoch - 1)
        
        # Log to console
        metrics_str = (
            f"\nEpoch {epoch + 1}/{self.config.num_epochs} | "
            f"Time {time_elapsed:.0f}s (ETA: {time_remaining:.0f}s)\n"
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f}\n"
            f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.4f}\n"
            f"GPU Memory: {train_metrics.get('train_gpu_memory', 0):.2f} GB"
        )
        self.logger.info(metrics_str)
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
