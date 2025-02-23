"""Evaluation utilities for sign language detection models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import json
from sklearn.metrics import confusion_matrix

from ..models import (
    SignLanguageCNNLSTM,
    VideoTransformer,
    CNNLSTMConfig,
    TransformerConfig
)
from ..data import create_dataloaders
from .metrics import calculate_metrics
from ..utils import get_checkpoint_dir

class ModelEvaluator:
    """Model evaluation handler."""
    
    def __init__(
        self,
        model_type: str,
        class_mapping: Dict[str, int],
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_type: One of ['cnn_lstm', 'transformer']
            class_mapping: Class name to index mapping
            checkpoint_dir: Directory containing checkpoints
            device: Device to evaluate on
        """
        self.model_type = model_type
        self.class_mapping = class_mapping
        self.checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._load_model()
    
    def _load_model(self) -> Union[SignLanguageCNNLSTM, VideoTransformer]:
        """Load model from checkpoint."""
        num_classes = len(self.class_mapping)
        
        # Create model
        if self.model_type == 'cnn_lstm':
            config = CNNLSTMConfig(num_classes=num_classes)
            model = SignLanguageCNNLSTM(config)
        elif self.model_type == 'transformer':
            config = TransformerConfig(num_classes=num_classes)
            model = VideoTransformer(config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_type}_final.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Evaluate single batch.
        
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
        
        # Calculate metrics
        metrics = calculate_metrics(
            class_pred,
            labels,
            bbox_pred,
            bboxes
        )
        
        return metrics
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Starting evaluation...")
        all_metrics = []
        all_preds = []
        all_labels = []
        
        for batch in test_loader:
            metrics = self.evaluate_batch(batch)
            all_metrics.append(metrics)
            
            # Store predictions for confusion matrix
            frames, (labels, _) = batch
            class_pred, _ = self.model(frames.to(self.device))
            all_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Add confusion matrix
        avg_metrics['confusion_matrix'] = confusion_matrix(
            all_labels,
            all_preds
        )
        
        return avg_metrics
    
    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        output_path: Optional[Path] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_mat: Confusion matrix array
            output_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.class_mapping.keys()),
            yticklabels=list(self.class_mapping.keys())
        )
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
    
    def plot_metric_history(
        self,
        history: Dict[str, List[float]],
        metric: str,
        output_path: Optional[Path] = None
    ):
        """
        Plot metric history.
        
        Args:
            history: Training history
            metric: Metric to plot
            output_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            history[f'train_{metric}'],
            label=f'Train {metric}'
        )
        plt.plot(
            history[f'val_{metric}'],
            label=f'Val {metric}'
        )
        plt.title(f'{metric.capitalize()} History - {self.model_type.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
    
    def save_results(
        self,
        metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ):
        """
        Save evaluation results.
        
        Args:
            metrics: Evaluation metrics
            output_path: Optional path to save results
        """
        if output_path is None:
            output_path = (
                self.checkpoint_dir /
                f"{self.model_type}_evaluation.json"
            )
        
        # Convert numpy types to Python types for JSON
        results = {}
        for k, v in metrics.items():
            if k == 'confusion_matrix':
                results[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                results[k] = float(v)
            else:
                results[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to {output_path}")

def main():
    """Run evaluation script."""
    import json
    from ..data import load_video_data
    
    # Load data
    video_data = load_video_data()
    
    # Load class mapping
    with open(Path(__file__).parent / 'class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    # Create data loaders
    _, _, test_loader = create_dataloaders(
        video_data=video_data,
        class_mapping=class_mapping
    )
    
    # Evaluate models
    for model_type in ['cnn_lstm', 'transformer']:
        evaluator = ModelEvaluator(
            model_type=model_type,
            class_mapping=class_mapping
        )
        
        # Run evaluation
        metrics = evaluator.evaluate(test_loader)
        
        # Save results
        evaluator.save_results(metrics)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            Path(f"{model_type}_confusion_matrix.png")
        )

if __name__ == '__main__':
    main()
