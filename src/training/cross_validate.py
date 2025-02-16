"""Cross-validation utilities for model evaluation."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import KFold
import logging
import json
from dataclasses import asdict

from ..models import (
    SignLanguageCNNLSTM,
    VideoTransformer,
    CNNLSTMConfig,
    TransformerConfig
)
from ..data import create_dataloaders
from .trainer import Trainer, TrainerConfig
from .metrics import calculate_metrics
from ..utils import get_checkpoint_dir

class CrossValidator:
    """K-fold cross-validation handler."""
    
    def __init__(
        self,
        model_type: str,
        video_data: List[Dict],
        class_mapping: Dict[str, int],
        n_splits: int = 5,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize cross-validator.
        
        Args:
            model_type: One of ['cnn_lstm', 'transformer']
            video_data: List of video data dictionaries
            class_mapping: Class name to index mapping
            n_splits: Number of folds
            checkpoint_dir: Directory to save checkpoints
            device: Device to run on
        """
        self.model_type = model_type
        self.video_data = video_data
        self.class_mapping = class_mapping
        self.n_splits = n_splits
        self.checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_model(
        self,
        config: Optional[Union[CNNLSTMConfig, TransformerConfig]] = None
    ) -> Union[SignLanguageCNNLSTM, VideoTransformer]:
        """Create model instance."""
        num_classes = len(self.class_mapping)
        
        if self.model_type == 'cnn_lstm':
            if config is None:
                config = CNNLSTMConfig(num_classes=num_classes)
            model = SignLanguageCNNLSTM(config)
        elif self.model_type == 'transformer':
            if config is None:
                config = TransformerConfig(num_classes=num_classes)
            model = VideoTransformer(config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def run_fold(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        fold: int,
        config: Optional[Union[CNNLSTMConfig, TransformerConfig]] = None,
        trainer_config: Optional[TrainerConfig] = None
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """
        Run single cross-validation fold.
        
        Args:
            train_data: Training data
            val_data: Validation data
            fold: Fold number
            config: Model configuration
            trainer_config: Training configuration
            
        Returns:
            Tuple of (training history, validation metrics)
        """
        self.logger.info(f"\nRunning fold {fold + 1}/{self.n_splits}")
        
        # Create model
        model = self._create_model(config)
        
        # Create data loaders
        train_loader, _, _ = create_dataloaders(
            train_data,
            self.class_mapping,
            train_split=1.0  # Use all data for training
        )
        _, val_loader, _ = create_dataloaders(
            val_data,
            self.class_mapping,
            train_split=0.0,  # Use all data for validation
            val_split=1.0
        )
        
        # Initialize trainer
        if trainer_config is None:
            trainer_config = TrainerConfig()
        
        trainer = Trainer(
            model=model,
            config=trainer_config
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on validation set
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for batch in val_loader:
                frames, (labels, bboxes) = batch
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)
                
                class_pred, bbox_pred = model(frames)
                metrics = calculate_metrics(
                    class_pred,
                    labels,
                    bbox_pred,
                    bboxes
                )
                
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
        
        # Average metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Save fold model
        fold_path = self.checkpoint_dir / f"{self.model_type}_fold_{fold + 1}.pth"
        trainer.save_checkpoint(fold_path)
        
        return history, val_metrics
    
    def cross_validate(
        self,
        config: Optional[Union[CNNLSTMConfig, TransformerConfig]] = None,
        trainer_config: Optional[TrainerConfig] = None
    ) -> Dict:
        """
        Run k-fold cross-validation.
        
        Args:
            config: Model configuration
            trainer_config: Training configuration
            
        Returns:
            Cross-validation results
        """
        self.logger.info(
            f"Starting {self.n_splits}-fold cross-validation "
            f"for {self.model_type.upper()}"
        )
        
        # Create folds
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        
        # Store results
        results = {
            'model_type': self.model_type,
            'n_splits': self.n_splits,
            'config': asdict(config) if config else None,
            'trainer_config': asdict(trainer_config) if trainer_config else None,
            'folds': []
        }
        
        # Run folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.video_data)):
            train_data = [self.video_data[i] for i in train_idx]
            val_data = [self.video_data[i] for i in val_idx]
            
            history, metrics = self.run_fold(
                train_data,
                val_data,
                fold,
                config,
                trainer_config
            )
            
            # Store fold results
            fold_results = {
                'fold': fold + 1,
                'history': history,
                'metrics': metrics
            }
            results['folds'].append(fold_results)
            
            self.logger.info(f"\nFold {fold + 1} Results:")
            for k, v in metrics.items():
                self.logger.info(f"{k}: {v:.4f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in results['folds'][0]['metrics'].keys():
            values = [
                fold['metrics'][metric]
                for fold in results['folds']
            ]
            avg_metrics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        results['average_metrics'] = avg_metrics
        
        # Log final results
        self.logger.info("\nCross-validation Results:")
        for metric, stats in avg_metrics.items():
            self.logger.info(
                f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}"
            )
        
        # Save results
        output_path = (
            self.checkpoint_dir /
            f"{self.model_type}_cross_validation.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nSaved results to {output_path}")
        
        return results

def main():
    """Run cross-validation."""
    import json
    from ..data import load_video_data
    
    # Load data
    video_data = load_video_data()
    
    # Load class mapping
    class_mapping_path = Path(__file__).parent / 'class_mapping.json'
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Run cross-validation for both models
    for model_type in ['cnn_lstm', 'transformer']:
        validator = CrossValidator(
            model_type=model_type,
            video_data=video_data,
            class_mapping=class_mapping
        )
        validator.cross_validate()

if __name__ == '__main__':
    main()
