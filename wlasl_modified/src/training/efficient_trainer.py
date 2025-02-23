"""Memory-efficient training implementation for sign language recognition."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import gc
import psutil
from tqdm import tqdm

from ..models.efficient_sign_net import EfficientSignNet
from ..config.model_config import MemoryOptimizedTrainingConfig, EfficientSignNetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryTracker:
    """Track and manage memory usage during training."""
    
    def __init__(self, warning_threshold: float = 0.9, critical_threshold: float = 0.95):
        """
        Initialize memory tracker.
        
        Args:
            warning_threshold: Memory usage warning threshold
            critical_threshold: Memory usage critical threshold
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_stats = []
    
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage."""
        stats = {
            'ram_used': psutil.Process().memory_info().rss / 1024**3,  # GB
            'ram_percent': psutil.Process().memory_percent(),
            'gpu_allocated': 0,
            'gpu_cached': 0
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_cached': torch.cuda.memory_reserved() / 1024**3
            })
        
        self.memory_stats.append(stats)
        return stats
    
    def should_cleanup(self) -> bool:
        """Determine if memory cleanup is needed."""
        stats = self.check_memory()
        
        if stats['ram_percent'] > self.critical_threshold * 100:
            return True
            
        if torch.cuda.is_available():
            gpu_utilization = stats['gpu_allocated'] / torch.cuda.get_device_properties(0).total_memory
            if gpu_utilization > self.critical_threshold:
                return True
                
        return False
    
    def cleanup(self):
        """Perform memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class EfficientTrainer:
    """Memory-efficient trainer for sign language recognition."""
    
    def __init__(self,
                 model: EfficientSignNet,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: MemoryOptimizedTrainingConfig,
                 device: torch.device,
                 checkpoint_dir: Path):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer with memory optimization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.num_epochs // 3,
            T_mult=2,
            eta_min=config.min_lr
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(
            warning_threshold=config.memory_warning_threshold,
            critical_threshold=config.memory_critical_threshold
        )
        
        # Enable gradient checkpointing if configured
        if config.use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with memory optimizations.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics = {'loss': 0.0, 'accuracy': 0.0}
        num_batches = len(self.train_loader)
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        accumulated_loss = 0
        
        with tqdm(total=num_batches, desc='Training') as pbar:
            for batch_idx, (frames, labels) in enumerate(self.train_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision forward pass
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
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
                if self.config.enable_memory_tracking and batch_idx % self.config.track_memory_every_n_steps == 0:
                    if self.memory_tracker.should_cleanup():
                        self.memory_tracker.cleanup()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': metrics['loss'] / (batch_idx + 1),
                    'acc': metrics['accuracy'] / (batch_idx + 1)
                })
        
        # Average metrics
        metrics['loss'] /= (num_batches // self.config.gradient_accumulation_steps)
        metrics['accuracy'] /= (num_batches // self.config.gradient_accumulation_steps)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model with memory optimization.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = {'val_loss': 0.0, 'val_accuracy': 0.0}
        num_batches = len(self.val_loader)
        
        for frames, labels in tqdm(self.val_loader, desc='Validation'):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision inference
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
            
            # Update metrics
            metrics['val_loss'] += loss.item()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item()
            metrics['val_accuracy'] += acc
            
            # Memory cleanup if needed
            if self.memory_tracker.should_cleanup():
                self.memory_tracker.cleanup()
        
        # Average metrics
        metrics['val_loss'] /= num_batches
        metrics['val_accuracy'] /= num_batches
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics['val_accuracy'] > getattr(self, 'best_val_accuracy', 0):
            self.best_val_accuracy = metrics['val_accuracy']
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Train model with memory optimizations.
        
        Args:
            num_epochs: Optional override for number of epochs
            
        Returns:
            Dictionary of training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                train_metrics = self.train_epoch()
                
                # Validation phase
                val_metrics = self.validate()
                
                # Update learning rate
                self.scheduler.step(val_metrics['val_loss'])
                
                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_accuracy'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                # Save checkpoint
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
                
                # Log progress
                logger.info(
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                )
                
                # Early stopping
                if self._should_stop_early(history):
                    logger.info("Early stopping triggered")
                    break
                
                # Memory cleanup between epochs
                self.memory_tracker.cleanup()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
        finally:
            # Final memory cleanup
            self.memory_tracker.cleanup()
        
        return history
    
    def _should_stop_early(self, history: Dict[str, list]) -> bool:
        """Check early stopping conditions."""
        if len(history['val_loss']) < self.config.early_stopping_patience:
            return False
            
        recent_losses = history['val_loss'][-self.config.early_stopping_patience:]
        min_loss = min(recent_losses)
        current_loss = recent_losses[-1]
        
        return (current_loss - min_loss) > self.config.early_stopping_min_delta

def create_trainer(
    model: EfficientSignNet,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: MemoryOptimizedTrainingConfig,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None
) -> EfficientTrainer:
    """
    Create memory-efficient trainer instance.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Optional device specification
        checkpoint_dir: Optional checkpoint directory
        
    Returns:
        Initialized trainer
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if checkpoint_dir is None:
        checkpoint_dir = Path('checkpoints/efficient_sign_net')
        
    return EfficientTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )