"""Tests for training components."""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import shutil
import tempfile
from typing import Dict

from src.training.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler
)
from src.training.metrics import (
    calculate_accuracy,
    calculate_iou,
    calculate_confusion_matrix,
    calculate_metrics
)

class DummyModel(nn.Module):
    """Dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

class DummyTrainer:
    """Dummy trainer for testing callbacks."""
    
    def __init__(self):
        """Initialize dummy trainer."""
        self.model = DummyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.history = {'train_loss': [], 'val_loss': []}
    
    def save_checkpoint(self, path: Path):
        """Mock checkpoint saving."""
        path.touch()

class TestCallbacks(unittest.TestCase):
    """Test cases for training callbacks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.trainer = DummyTrainer()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_model_checkpoint(self):
        """Test model checkpointing callback."""
        callback = ModelCheckpoint(
            dirpath=self.test_dir,
            monitor='val_loss',
            mode='min',
            save_top_k=2
        )
        
        # Test improving metrics
        metrics = {'val_loss': 1.0}
        callback(self.trainer, metrics)
        self.assertEqual(len(list(self.test_dir.glob('*.pth'))), 1)
        
        # Test non-improving metrics
        metrics = {'val_loss': 2.0}
        callback(self.trainer, metrics)
        self.assertEqual(len(list(self.test_dir.glob('*.pth'))), 1)
        
        # Test new best metrics
        metrics = {'val_loss': 0.5}
        callback(self.trainer, metrics)
        self.assertEqual(len(list(self.test_dir.glob('*.pth'))), 2)
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        callback = EarlyStopping(
            monitor='val_loss',
            patience=2,
            mode='min'
        )
        
        # Test non-stopping case
        metrics = {'val_loss': 1.0}
        self.assertFalse(callback(self.trainer, metrics))
        metrics = {'val_loss': 0.8}
        self.assertFalse(callback(self.trainer, metrics))
        
        # Test stopping case
        metrics = {'val_loss': 1.0}
        self.assertFalse(callback(self.trainer, metrics))
        metrics = {'val_loss': 1.1}
        self.assertFalse(callback(self.trainer, metrics))
        metrics = {'val_loss': 1.2}
        self.assertTrue(callback(self.trainer, metrics))
    
    def test_lr_scheduler(self):
        """Test learning rate scheduler callback."""
        initial_lr = 0.1
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = initial_lr
        
        callback = LearningRateScheduler(
            optimizer=self.trainer.optimizer,
            patience=2,
            factor=0.1
        )
        
        # Test non-reduction case
        metrics = {'val_loss': 1.0}
        callback(self.trainer, metrics)
        self.assertEqual(
            self.trainer.optimizer.param_groups[0]['lr'],
            initial_lr
        )
        
        # Test reduction case
        metrics = {'val_loss': 1.1}
        callback(self.trainer, metrics)
        metrics = {'val_loss': 1.2}
        callback(self.trainer, metrics)
        metrics = {'val_loss': 1.3}
        callback(self.trainer, metrics)
        
        self.assertEqual(
            self.trainer.optimizer.param_groups[0]['lr'],
            initial_lr * 0.1
        )

class TestMetrics(unittest.TestCase):
    """Test cases for training metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        pred = torch.tensor([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ]).to(self.device)
        
        target = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]).to(self.device)
        
        accuracy, pred_classes = calculate_accuracy(pred, target)
        self.assertIsInstance(accuracy, float)
        self.assertEqual(accuracy, 1.0)  # All predictions correct
    
    def test_iou_calculation(self):
        """Test IoU calculation."""
        pred_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0]
        ]).to(self.device)
        
        target_boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5]
        ]).to(self.device)
        
        iou = calculate_iou(pred_boxes, target_boxes)
        self.assertIsInstance(iou, float)
        self.assertTrue(0 <= iou <= 1)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        pred = torch.tensor([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ]).to(self.device)
        
        target = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]).to(self.device)
        
        confusion_mat = calculate_confusion_matrix(pred, target, num_classes=2)
        self.assertEqual(confusion_mat.shape, (2, 2))
        self.assertEqual(confusion_mat.sum().item(), 3)  # Total samples
    
    def test_metrics_calculation(self):
        """Test full metrics calculation."""
        class_pred = torch.tensor([
            [0.1, 0.9],
            [0.8, 0.2]
        ]).to(self.device)
        
        class_target = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0]
        ]).to(self.device)
        
        bbox_pred = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0]
        ]).to(self.device)
        
        bbox_target = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5]
        ]).to(self.device)
        
        metrics = calculate_metrics(
            class_pred,
            class_target,
            bbox_pred,
            bbox_target
        )
        
        required_metrics = [
            'accuracy',
            'iou',
            'mean_precision',
            'mean_recall',
            'mean_f1'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)

def main():
    """Run the tests."""
    unittest.main()

if __name__ == '__main__':
    main()
