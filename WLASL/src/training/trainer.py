"""Memory-efficient model training with cross-validation support."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

from ...configs.base_config import (
    TRAIN_CONFIG, 
    CHECKPOINTS_DIR,
    LOG_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """Memory-efficient trainer supporting I3D and TGCN models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 fold_idx: Optional[int] = None):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fold_idx = fold_idx
        
        # Training utilities
        self.scaler = GradScaler() if TRAIN_CONFIG['mixed_precision'] else None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = LOG_DIR / f'fold_{fold_idx}' if fold_idx is not None else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader)):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Mixed precision training
            with autocast(enabled=TRAIN_CONFIG['mixed_precision']):
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            # Scale loss and compute gradients
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if TRAIN_CONFIG['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        TRAIN_CONFIG['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if TRAIN_CONFIG['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        TRAIN_CONFIG['gradient_clip']
                    )
                    
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
        
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                with autocast(enabled=TRAIN_CONFIG['mixed_precision']):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = TRAIN_CONFIG['num_epochs']) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            
            # Validation phase
            val_loss, val_acc = self.validate()
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                logger.info('Early stopping triggered')
                break
                
            # Save training history
            self.save_history()
            
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        fold_str = f'_fold_{self.fold_idx}' if self.fold_idx is not None else ''
        checkpoint_path = CHECKPOINTS_DIR / f'checkpoint_{self.run_id}{fold_str}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path}')
        
    def save_history(self):
        """Save training history."""
        history_path = self.log_dir / f'history_{self.run_id}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)