"""Training callbacks for model training."""

import torch
from torch.optim import Optimizer
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

class ModelCheckpoint:
    """Save model checkpoints based on monitored metric."""
    
    def __init__(
        self,
        dirpath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 1,
        filename: str = 'checkpoint_{epoch:02d}_{monitor:.4f}'
    ):
        """
        Initialize callback.
        
        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor
            mode: One of ['min', 'max']
            save_top_k: Number of best models to save
            filename: Checkpoint filename pattern
        """
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        
        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.best_k_models: Dict[Path, float] = {}
        self.best_k_scores = []
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best
        return current > best
    
    def format_checkpoint_name(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Format checkpoint filename."""
        filename = self.filename
        filename = filename.replace('{epoch}', f'{epoch:02d}')
        filename = filename.replace('{monitor}', f'{metrics[self.monitor]:.4f}')
        return filename + '.pth'
    
    def __call__(
        self,
        trainer: 'Trainer',  # type: ignore # Forward reference
        metrics: Dict[str, float]
    ) -> bool:
        """
        Save checkpoint if conditions are met.
        
        Args:
            trainer: Model trainer
            metrics: Current metrics
            
        Returns:
            False (callback never triggers early stopping)
        """
        current = metrics[self.monitor]
        epoch = len(trainer.history['train_loss'])
        
        # Check if current model is better
        if self.is_better(current, self.best_score):
            self.best_score = current
            
            # Save checkpoint
            filename = self.format_checkpoint_name(epoch, metrics)
            filepath = self.dirpath / filename
            trainer.save_checkpoint(filepath)
            
            # Update best k models
            self.best_k_models[filepath] = current
            self.best_k_scores.append(current)
            
            # Sort and prune old models
            if self.save_top_k > 0:
                _scores = torch.tensor(self.best_k_scores)
                _paths = list(self.best_k_models.keys())
                
                # Sort scores and corresponding paths
                sorted_indices = (
                    torch.argsort(_scores)
                    if self.mode == 'min'
                    else torch.argsort(_scores, descending=True)
                )
                
                # Remove excess models
                for idx in range(len(_paths)):
                    if idx >= self.save_top_k:
                        _paths[sorted_indices[idx]].unlink(missing_ok=True)
                        del self.best_k_models[_paths[sorted_indices[idx]]]
            
            self.logger.info(
                f"Saved checkpoint: {filepath} "
                f"({self.monitor}: {current:.4f})"
            )
        
        return False  # Never triggers early stopping

class EarlyStopping:
    """Stop training when monitored metric stops improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize callback.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as an improvement
            patience: Number of epochs to wait for improvement
            mode: One of ['min', 'max']
            verbose: Whether to log messages
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        
        # Initialize state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)
    
    def __call__(
        self,
        trainer: 'Trainer',  # type: ignore # Forward reference
        metrics: Dict[str, float]
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            trainer: Model trainer
            metrics: Current metrics
            
        Returns:
            True if training should stop
        """
        current = metrics[self.monitor]
        
        if self.is_better(current, self.best_score):
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = len(trainer.history['train_loss'])
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered: no improvement in "
                        f"{self.monitor} for {self.patience} epochs"
                    )
                return True
        
        return False

class LearningRateScheduler:
    """Adjust learning rate based on monitored metric."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        """
        Initialize callback.
        
        Args:
            optimizer: Model optimizer
            mode: One of ['min', 'max']
            factor: Factor to reduce learning rate by
            patience: Number of epochs to wait for improvement
            min_lr: Minimum learning rate
            verbose: Whether to log messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Initialize state
        self.wait = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best
        return current > best
    
    def __call__(
        self,
        trainer: 'Trainer',  # type: ignore # Forward reference
        metrics: Dict[str, float]
    ) -> bool:
        """
        Adjust learning rate if needed.
        
        Args:
            trainer: Model trainer
            metrics: Current metrics
            
        Returns:
            False (callback never triggers early stopping)
        """
        current = metrics['val_loss']  # Always use validation loss
        
        if self.is_better(current, self.best_score):
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    if self.verbose:
                        self.logger.info(
                            f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}"
                        )
                
                self.wait = 0  # Reset counter
        
        return False  # Never triggers early stopping
