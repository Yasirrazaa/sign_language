"""Cross-validation implementation for sign language detection."""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
import logging
from pathlib import Path
import json

from .trainer import Trainer, TrainerConfig
from ..models import SignLanguageCNNLSTM, VideoTransformer
from ..config import TRAIN_CONFIG
from .metrics import calculate_metrics

class CrossValidator:
    """K-fold cross-validation implementation."""
    
    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        trainer_config: TrainerConfig,
        num_folds: int = TRAIN_CONFIG['num_folds'],
        shuffle: bool = True,
        random_seed: int = TRAIN_CONFIG['random_seed']
    ):
        """
        Initialize cross-validator.
        
        Args:
            model_class: Model class (CNNLSTMConfig or TransformerConfig)
            model_config: Model configuration
            trainer_config: Trainer configuration
            num_folds: Number of folds
            shuffle: Whether to shuffle data
            random_seed: Random seed
        """
        self.model_class = model_class
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.num_folds = num_folds
        self.random_seed = random_seed
        
        # Initialize k-fold splitter
        self.kfold = KFold(
            n_splits=num_folds,
            shuffle=shuffle,
            random_state=random_seed
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Store results
        self.fold_histories = []
        self.fold_metrics = []
    
    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_dir: Path
    ) -> Tuple[Dict, Dict]:
        """
        Train and evaluate a single fold.
        
        Args:
            fold: Fold number
            train_loader: Training data loader
            val_loader: Validation data loader
            log_dir: Directory to save logs
            
        Returns:
            Tuple of (training history, final metrics)
        """
        self.logger.info(f"\nTraining Fold {fold + 1}/{self.num_folds}")
        
        # Initialize model
        model = self.model_class(self.model_config).to(self.trainer_config.device)
        
        # Initialize trainer
        trainer = Trainer(
            model,
            self.trainer_config,
            checkpoint_dir=log_dir / f"fold_{fold + 1}"
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Get final metrics on validation set
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader:
                frames, labels = batch
                frames = frames.to(self.trainer_config.device)
                labels = torch.argmax(labels, dim=1).to(self.trainer_config.device)
                
                predictions = model(frames)
                batch_metrics = calculate_metrics(predictions, labels)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
        
        # Average metrics
        final_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Save fold results
        results = {
            'fold': fold + 1,
            'history': history,
            'final_metrics': final_metrics
        }
        with open(log_dir / f"fold_{fold + 1}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return history, final_metrics
    
    def cross_validate(
        self,
        dataset,
        batch_size: int,
        log_dir: Path
    ) -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            dataset: Full dataset
            batch_size: Batch size for dataloaders
            log_dir: Directory to save logs
            
        Returns:
            Dictionary of aggregated results
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get indices for all samples
        indices = list(range(len(dataset)))
        
        # Perform k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            # Create data samplers
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                pin_memory=True
            )
            val_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                pin_memory=True
            )
            
            # Train and evaluate fold
            history, metrics = self.train_fold(fold, train_loader, val_loader, log_dir)
            
            # Store results
            self.fold_histories.append(history)
            self.fold_metrics.append(metrics)
            
            # Log fold results
            self._log_fold_results(fold, metrics)
        
        # Calculate and log aggregate results
        aggregate_results = self._aggregate_results()
        self._log_aggregate_results(aggregate_results)
        
        # Save aggregate results
        with open(log_dir / "aggregate_results.json", 'w') as f:
            json.dump(aggregate_results, f, indent=2)
        
        return aggregate_results
    
    def _log_fold_results(self, fold: int, metrics: Dict):
        """Log results for a single fold."""
        self.logger.info(f"\nFold {fold + 1} Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
    
    def _log_aggregate_results(self, results: Dict):
        """Log aggregate results across all folds."""
        self.logger.info("\nAggregate Results:")
        self.logger.info("-" * 50)
        
        for metric, stats in results.items():
            self.logger.info(f"\n{metric}:")
            self.logger.info(f"  Mean:   {stats['mean']:.4f}")
            self.logger.info(f"  Std:    {stats['std']:.4f}")
            self.logger.info(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    def _aggregate_results(self) -> Dict:
        """
        Calculate aggregate statistics across all folds.
        
        Returns:
            Dictionary of aggregate statistics for each metric
        """
        # Get all metrics
        metrics = self.fold_metrics[0].keys()
        
        # Calculate statistics for each metric
        results = {}
        for metric in metrics:
            values = [fold[metric] for fold in self.fold_metrics]
            mean = np.mean(values)
            std = np.std(values)
            
            # Calculate 95% confidence interval
            ci = 1.96 * std / np.sqrt(self.num_folds)
            
            results[metric] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(mean - ci),
                'ci_upper': float(mean + ci),
                'values': values
            }
        
        return results
