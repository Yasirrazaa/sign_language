"""Cross-validation with integrated data analysis and evaluation."""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from ..data.data_loader import create_data_loaders
from .trainer import Trainer
from ...configs.base_config import (
    TRAIN_CONFIG,
    EVAL_CONFIG,
    ANALYSIS_DIR,
    LOG_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossValidator:
    """Cross-validation with comprehensive analysis and evaluation."""
    
    def __init__(self,
                 model_class: nn.Module,
                 model_params: Dict,
                 data_info: List[Dict],
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 num_folds: int = TRAIN_CONFIG['num_folds']):
        """
        Initialize cross-validator.
        
        Args:
            model_class: Model class (I3D or TGCN)
            model_params: Model parameters
            data_info: List of dictionaries containing video metadata
            criterion: Loss function
            num_folds: Number of cross-validation folds
        """
        self.model_class = model_class
        self.model_params = model_params
        self.data_info = data_info
        self.criterion = criterion
        self.num_folds = num_folds
        
        # Setup directories
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = ANALYSIS_DIR / f'cv_results_{self.run_id}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.fold_results = []
        self.aggregate_metrics = {}
        
    def run(self) -> Dict:
        """
        Run cross-validation with analysis.
        
        Returns:
            Dictionary containing all results and metrics
        """
        logger.info(f"Starting {self.num_folds}-fold cross-validation")
        
        for fold_idx in range(self.num_folds):
            logger.info(f"Training fold {fold_idx + 1}/{self.num_folds}")
            
            # Create data loaders for this fold
            dataloaders = create_data_loaders(
                self.data_info,
                fold_idx=fold_idx
            )
            
            # Initialize model
            model = self.model_class(**self.model_params)
            
            # Setup optimizer and scheduler
            optimizer = Adam(
                model.parameters(),
                lr=TRAIN_CONFIG['learning_rate'],
                weight_decay=TRAIN_CONFIG['weight_decay']
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=TRAIN_CONFIG['reduce_lr_factor'],
                patience=TRAIN_CONFIG['reduce_lr_patience'],
                min_lr=TRAIN_CONFIG['min_learning_rate']
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                criterion=self.criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                fold_idx=fold_idx
            )
            
            # Train the model
            history = trainer.train()
            
            # Evaluate on test set
            test_metrics = self.evaluate_fold(
                model,
                dataloaders['test'],
                fold_idx
            )
            
            # Store results
            fold_results = {
                'history': history,
                'test_metrics': test_metrics
            }
            self.fold_results.append(fold_results)
            
            # Save fold results
            self.save_fold_results(fold_idx, fold_results)
            
        # Aggregate and analyze results
        self.analyze_results()
        
        return {
            'fold_results': self.fold_results,
            'aggregate_metrics': self.aggregate_metrics
        }
    
    def evaluate_fold(self,
                     model: nn.Module,
                     test_loader: torch.utils.data.DataLoader,
                     fold_idx: int) -> Dict:
        """Evaluate model on test set for current fold."""
        model.eval()
        device = next(model.parameters()).device
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data = data.to(device)
                output = model(data)
                
                # Get predictions
                pred = output.max(1)[1].cpu().numpy()
                predictions.extend(pred)
                targets.extend(target.numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Classification report
        report = classification_report(
            targets,
            predictions,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Top-k accuracy
        top_k_acc = self.calculate_top_k_accuracy(
            model,
            test_loader,
            k_values=EVAL_CONFIG['top_k']
        )
        
        metrics = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'top_k_accuracy': top_k_acc
        }
        
        # Save visualizations
        self.plot_confusion_matrix(
            conf_matrix,
            fold_idx,
            save_dir=self.results_dir
        )
        
        return metrics
    
    def calculate_top_k_accuracy(self,
                               model: nn.Module,
                               data_loader: torch.utils.data.DataLoader,
                               k_values: List[int]) -> Dict[int, float]:
        """Calculate top-k accuracy for specified k values."""
        model.eval()
        device = next(model.parameters()).device
        
        top_k_correct = {k: 0 for k in k_values}
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                
                # Calculate top-k accuracy
                _, pred = output.topk(max(k_values), 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                
                # Update counters
                total += target.size(0)
                for k in k_values:
                    top_k_correct[k] += correct[:k].reshape(-1).float().sum(0)
        
        # Calculate accuracies
        return {k: (correct/total).item()*100 for k, correct in top_k_correct.items()}
    
    def analyze_results(self):
        """Analyze cross-validation results."""
        # Aggregate metrics across folds
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold_results in self.fold_results:
            report = fold_results['test_metrics']['classification_report']
            all_metrics['accuracy'].append(report['accuracy'])
            all_metrics['precision'].append(report['macro avg']['precision'])
            all_metrics['recall'].append(report['macro avg']['recall'])
            all_metrics['f1'].append(report['macro avg']['f1-score'])
        
        # Calculate mean and std for each metric
        self.aggregate_metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in all_metrics.items()
        }
        
        # Plot learning curves
        self.plot_learning_curves()
        
        # Save aggregate results
        self.save_aggregate_results()
    
    def plot_confusion_matrix(self,
                            conf_matrix: np.ndarray,
                            fold_idx: int,
                            save_dir: Path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        plt.savefig(save_dir / f'confusion_matrix_fold_{fold_idx}.png')
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves across all folds."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for fold_idx, results in enumerate(self.fold_results):
            history = results['history']
            
            # Loss curves
            ax1.plot(history['train_loss'], label=f'Fold {fold_idx + 1} - Train')
            ax1.plot(history['val_loss'], label=f'Fold {fold_idx + 1} - Val')
            
            # Accuracy curves
            ax2.plot(history['train_acc'], label=f'Fold {fold_idx + 1} - Train')
            ax2.plot(history['val_acc'], label=f'Fold {fold_idx + 1} - Val')
        
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'learning_curves.png')
        plt.close()
    
    def save_fold_results(self, fold_idx: int, results: Dict):
        """Save results for individual fold."""
        save_path = self.results_dir / f'fold_{fold_idx}_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self.make_json_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def save_aggregate_results(self):
        """Save aggregate results and metrics."""
        save_path = self.results_dir / 'aggregate_results.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.aggregate_metrics, f, indent=4)
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: CrossValidator.make_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [CrossValidator.make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj