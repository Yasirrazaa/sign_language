"""Memory-efficient cross-validation implementation."""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from typing import Dict, List, Type, Union
from sklearn.model_selection import KFold
import logging
from pathlib import Path
import json
import gc

from .trainer import MemoryEfficientTrainer, TrainerConfig
from ..models import VideoTransformer, CNNTransformer, TimeSformer, I3DTransformer
from ..config import TRAIN_CONFIG
from .metrics import calculate_metrics

class MemoryEfficientCrossValidator:
    """Memory-efficient K-fold cross-validation implementation."""
    
    def __init__(
        self,
        model_class: Type[Union[VideoTransformer, CNNTransformer, TimeSformer, I3DTransformer]],
        model_config: Dict,
        trainer_config: TrainerConfig,
        num_folds: int = TRAIN_CONFIG['num_folds'],
        shuffle: bool = True,
        random_seed: int = TRAIN_CONFIG['random_seed']
    ):
        """
        Initialize cross-validator.
        
        Args:
            model_class: Model class to validate
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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store results
        self.fold_histories = []
        self.fold_metrics = []
    
    def _clear_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_dir: Path
    ) -> Dict:
        """Train and evaluate a single fold."""
        self.logger.info(f"\nTraining Fold {fold + 1}/{self.num_folds}")
        
        try:
            # Initialize model
            model = self.model_class(**self.model_config)
            model = model.to(self.trainer_config.device)
            
            # Initialize trainer
            trainer = MemoryEfficientTrainer(
                model,
                self.trainer_config
            )
            trainer.checkpoint_dir = log_dir / f"fold_{fold + 1}"
            
            # Train model
            history = trainer.train(train_loader, val_loader)
            
            # Final validation
            val_metrics = trainer.validate_epoch(val_loader)
            
            # Save fold results
            results = {
                'fold': fold + 1,
                'history': history,
                'metrics': val_metrics
            }
            
            results_path = log_dir / f"fold_{fold + 1}_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        finally:
            # Clean up memory
            self._clear_memory()
    
    def cross_validate(
        self,
        dataset,
        batch_size: int,
        log_dir: Path
    ) -> Dict:
        """Perform k-fold cross-validation."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get indices for all samples
            indices = list(range(len(dataset)))
            
            # Perform k-fold cross validation
            for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
                # Create data samplers
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                
                # Create data loaders
                loader_kwargs = {
                    'batch_size': batch_size,
                    'pin_memory': True,
                    'num_workers': 4,
                    'prefetch_factor': 2
                }
                
                train_loader = DataLoader(
                    dataset,
                    sampler=train_sampler,
                    **loader_kwargs
                )
                val_loader = DataLoader(
                    dataset,
                    sampler=val_sampler,
                    **loader_kwargs
                )
                
                # Train and evaluate fold
                results = self.train_fold(fold, train_loader, val_loader, log_dir)
                
                # Store results
                self.fold_histories.append(results['history'])
                self.fold_metrics.append(results['metrics'])
                
                # Log fold results
                self._log_fold_results(fold, results['metrics'])
                
                # Clear memory between folds
                self._clear_memory()
            
            # Calculate and log aggregate results
            aggregate_results = self._aggregate_results()
            self._log_aggregate_results(aggregate_results)
            
            # Save aggregate results
            with open(log_dir / "aggregate_results.json", 'w') as f:
                json.dump(aggregate_results, f, indent=2)
            
            return aggregate_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise
        finally:
            # Final cleanup
            self._clear_memory()
    
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
        """Calculate aggregate statistics across all folds."""
        metrics = self.fold_metrics[0].keys()
        
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
