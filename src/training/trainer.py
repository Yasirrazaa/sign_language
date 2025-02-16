"""Training implementation for sign language detection models."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
from dataclasses import dataclass

from ..models import SignLanguageCNNLSTM, VideoTransformer
from ..utils import get_checkpoint_dir
from .callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from .metrics import calculate_metrics

@dataclass
class TrainerConfig:
    """Configuration for training."""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    class_loss_weight: float = 1.0
    bbox_loss_weight: float = 1.0
    use_wandb: bool = True
    checkpoint_dir: Optional[Path] = None
    device: Optional[torch.device] = None

class Trainer:
    """Model trainer implementation."""
    
    def __init__(
        self,
        model: Union[SignLanguageCNNLSTM, VideoTransformer],
        config: TrainerConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        
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
        
        # Initialize loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.MSELoss()
        
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
            'train_iou': [], 'val_iou': [],
            'train_class_loss': [], 'val_class_loss': [],
            'train_bbox_loss': [], 'val_bbox_loss': [],
            'train_mean_precision': [], 'val_mean_precision': [],
            'train_mean_recall': [], 'val_mean_recall': [],
            'train_mean_f1': [], 'val_mean_f1': [],
            'learning_rate': []
        }
        
        # Initialize callbacks
        self.callbacks = self._setup_callbacks()
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._setup_wandb()
    
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
                patience=10,
                mode='min'
            )
        )
        
        # Learning rate scheduling
        callbacks.append(
            LearningRateScheduler(
                optimizer=self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
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
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Tuple of (frames, (labels, bboxes))
            
        Returns:
            Dictionary of metrics
        """
        frames, (labels, bboxes) = batch
        frames = frames.to(self.device)
        labels = labels.to(self.device)
        bboxes = bboxes.to(self.device)
        
        # Forward pass
        class_pred, bbox_pred = self.model(frames)
        
        # Calculate losses
        class_loss = self.class_criterion(class_pred, labels)
        bbox_loss = self.bbox_criterion(bbox_pred, bboxes)
        
        # Combined loss
        loss = (
            self.config.class_loss_weight * class_loss +
            self.config.bbox_loss_weight * bbox_loss
        )
        
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
        
        # Calculate metrics
        metrics = calculate_metrics(
            class_pred.detach(),
            labels.detach(),
            bbox_pred.detach(),
            bboxes.detach()
        )
        metrics.update({
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'bbox_loss': bbox_loss.item()
        })
        
        # Prefix metrics with 'train_'
        train_metrics = {f'train_{k}': v for k, v in metrics.items()}
        
        return train_metrics
    
    @torch.no_grad()
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Tuple of (frames, (labels, bboxes))
            
        Returns:
            Dictionary of metrics prefixed with 'val_'
        """
        frames, (labels, bboxes) = batch
        frames = frames.to(self.device)
        labels = labels.to(self.device)
        bboxes = bboxes.to(self.device)
        
        # Forward pass
        class_pred, bbox_pred = self.model(frames)
        
        # Calculate losses
        class_loss = self.class_criterion(class_pred, labels)
        bbox_loss = self.bbox_criterion(bbox_pred, bboxes)
        
        # Combined loss
        loss = (
            self.config.class_loss_weight * class_loss +
            self.config.bbox_loss_weight * bbox_loss
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            class_pred,
            labels,
            bbox_pred,
            bboxes
        )
        metrics.update({
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'bbox_loss': bbox_loss.item()
        })
        
        # Prefix metrics with 'val_'
        val_metrics = {f'val_{k}': v for k, v in metrics.items()}
        
        return val_metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.train()
        epoch_metrics = []
        
        for batch in tqdm(train_loader, desc='Training'):
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
        
        # Average metrics (metrics are already prefixed with 'train_')
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.eval()
        epoch_metrics = []
        
        for batch in tqdm(val_loader, desc='Validating'):
            metrics = self.validate_step(batch)
            epoch_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            for key in epoch_metrics[0].keys()
        }
        
        return avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary of metric histories
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
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
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics for current epoch."""
        # Console logging
        self.logger.info("\nTraining metrics:")
        for k, v in train_metrics.items():
            self.logger.info(f"{k}: {v:.4f}")
        
        self.logger.info("\nValidation metrics:")
        for k, v in val_metrics.items():
            self.logger.info(f"{k}: {v:.4f}")
        
        # W&B logging
        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **val_metrics
            })
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
