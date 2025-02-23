"""Training callbacks for model training."""

import torch
from pathlib import Path
from typing import Dict, Optional
import numpy as np

class ModelCheckpoint:
    """Save best model checkpoints based on monitored metric."""
    
    def __init__(
        self,
        dirpath: Path,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best models to save
        """
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        
        self.best_k_models = {}
        self.metric_history = []
        
        # Initialize tracking variables
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Create checkpoint directory
        self.dirpath.mkdir(parents=True, exist_ok=True)
    
    def is_better(self, current: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < self.best_score
        return current > self.best_score
    
    def __call__(self, trainer, metrics: Dict[str, float]) -> bool:
        """
        Save checkpoint if monitored metric improved.
        
        Returns:
            bool: Whether training should be stopped
        """
        current = metrics[self.monitor]
        self.metric_history.append(current)
        
        # Update best score and save model if improved
        if self.is_better(current):
            self.best_score = current
            
            # Save checkpoint
            checkpoint_path = self.dirpath / f'model_best_{self.monitor}_{current:.4f}.pth'
            trainer.save_checkpoint(checkpoint_path)
            
            # Update best k models
            self.best_k_models[checkpoint_path] = current
            
            # Remove worst checkpoint if we have too many
            if len(self.best_k_models) > self.save_top_k:
                worst_path = min(self.best_k_models.items(), 
                               key=lambda x: x[1] if self.mode == 'max' else -x[1])[0]
                worst_path.unlink(missing_ok=True)
                del self.best_k_models[worst_path]
        
        return False  # Never triggers stopping

class EarlyStopping:
    """Stop training when monitored metric stops improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as an improvement
            patience: Number of epochs to wait for improvement
            mode: 'min' or 'max'
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        
        self.min_delta *= 1 if self.mode == 'min' else -1
    
    def is_better(self, current: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        return current > (self.best_score + self.min_delta)
    
    def __call__(self, trainer, metrics: Dict[str, float]) -> bool:
        """
        Check if training should be stopped.
        
        Returns:
            bool: Whether training should be stopped
        """
        current = metrics[self.monitor]
        
        if self.is_better(current):
            self.best_score = current
            self.counter = 0
            self.best_epoch = trainer.history['train_loss'].__len__()
        else:
            self.counter += 1
            
        # Log warning when getting close to stopping
        if self.counter > self.patience - 3:
            trainer.logger.warning(
                f'Early stopping counter: {self.counter} out of {self.patience}'
            )
        
        return self.counter >= self.patience

class WarmupScheduler:
    """Learning rate scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        initial_lr: float,
        min_lr: float
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            initial_lr: Target learning rate after warmup
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate based on current epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (100 - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
    
    def state_dict(self) -> Dict:
        """Get scheduler state."""
        return {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
