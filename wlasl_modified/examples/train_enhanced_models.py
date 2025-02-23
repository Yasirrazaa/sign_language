"""Training script for enhanced sign language recognition models with cross-validation."""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import yaml
import argparse
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import wandb

from src.models.enhanced_models import create_memory_efficient_model
from src.training.efficient_trainer import EfficientTrainer
from src.data.loader import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CrossValidationTrainer:
    """7-fold cross-validation trainer for enhanced models."""
    
    def __init__(self,
                 config_path: str,
                 data_dir: Path,
                 output_dir: Path,
                 model_type: str):
        """
        Initialize cross-validation trainer.
        
        Args:
            config_path: Path to configuration file
            data_dir: Data directory
            output_dir: Output directory
            model_type: Type of model ('cnn_lstm' or '3dcnn')
        """
        self.config = load_config(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup cross-validation
        self.kfold = KFold(
            n_splits=self.config['common']['cross_validation']['num_folds'],
            shuffle=True,
            random_state=42
        )
        
        # Initialize WandB
        wandb.init(
            project="sign-language-recognition",
            config=self.config,
            name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def train_fold(self,
                   fold: int,
                   train_indices: np.ndarray,
                   val_indices: np.ndarray) -> Dict:
        """Train model on a single fold."""
        logger.info(f"\nTraining Fold {fold + 1}")
        
        # Create data loaders for this fold
        data_loaders = create_data_loaders(
            self.data_dir,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=self.config['common']['cross_validation']['batch_size'],
            num_workers=self.config['hardware']['num_workers']
        )
        
        # Create model
        model = create_memory_efficient_model(
            model_type=self.model_type,
            num_classes=self.config['common']['num_classes'],
            input_shape=tuple(self.config['common']['input_shape'])
        )
        
        # Get model-specific config
        model_config = self.config[self.model_type]
        
        # Create trainer
        trainer = EfficientTrainer(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            config=model_config['training'],
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            checkpoint_dir=self.output_dir / f'fold_{fold}'
        )
        
        # Train for specified epochs
        history = trainer.train(
            num_epochs=self.config['common']['cross_validation']['epochs_per_fold']
        )
        
        # Log fold results
        wandb.log({
            f"fold_{fold}/best_val_accuracy": max(history['val_accuracy']),
            f"fold_{fold}/final_train_loss": history['train_loss'][-1]
        })
        
        return history
    
    def run_cross_validation(self) -> List[Dict]:
        """Run full cross-validation training."""
        histories = []
        all_data = np.arange(len(self.data_dir))  # Get all sample indices
        
        try:
            # Train each fold
            for fold, (train_idx, val_idx) in enumerate(self.kfold.split(all_data)):
                fold_history = self.train_fold(fold, train_idx, val_idx)
                histories.append(fold_history)
                
                # Save fold results
                self._save_fold_results(fold, fold_history)
            
            # Calculate and save aggregate results
            self._save_aggregate_results(histories)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise
            
        finally:
            wandb.finish()
        
        return histories
    
    def _save_fold_results(self, fold: int, history: Dict):
        """Save results for a single fold."""
        results_file = self.output_dir / f'fold_{fold}/results.yml'
        with open(results_file, 'w') as f:
            yaml.dump(history, f)
    
    def _save_aggregate_results(self, histories: List[Dict]):
        """Calculate and save aggregate results across all folds."""
        # Calculate mean and std of metrics
        metrics = {}
        for metric in ['val_accuracy', 'val_loss']:
            values = [max(h[metric]) if metric == 'val_accuracy' else min(h[metric]) 
                     for h in histories]
            metrics[f'mean_{metric}'] = float(np.mean(values))
            metrics[f'std_{metric}'] = float(np.std(values))
        
        # Save results
        results_file = self.output_dir / 'aggregate_results.yml'
        with open(results_file, 'w') as f:
            yaml.dump(metrics, f)
        
        # Log to WandB
        wandb.log({
            "final/mean_accuracy": metrics['mean_val_accuracy'],
            "final/std_accuracy": metrics['std_val_accuracy']
        })

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train enhanced sign language recognition models"
    )
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Path to output directory')
    parser.add_argument('--model-type', type=str, choices=['cnn_lstm', '3dcnn'],
                      required=True, help='Type of model to train')
    
    args = parser.parse_args()
    
    # Initialize and run cross-validation
    trainer = CrossValidationTrainer(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
    
    histories = trainer.run_cross_validation()
    
    # Print final results
    results_file = Path(args.output_dir) / 'aggregate_results.yml'
    with open(results_file, 'r') as f:
        results = yaml.safe_load(f)
    
    logger.info("\nFinal Results:")
    logger.info(f"Mean Validation Accuracy: {results['mean_val_accuracy']:.4f} ± "
               f"{results['std_val_accuracy']:.4f}")

if __name__ == '__main__':
    main()