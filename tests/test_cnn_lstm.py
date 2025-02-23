"""Test script for CNN-LSTM model implementation."""

import unittest
import torch
import numpy as np
from pathlib import Path

from models.cnn_lstm import SignLanguageCNNLSTM, CNNLSTMConfig
from src.config import MODEL_CONFIG, DATA_CONFIG

class TestSignLanguageCNNLSTM(unittest.TestCase):
    """Test cases for SignLanguageCNNLSTM model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = CNNLSTMConfig(
            num_classes=10,
            hidden_size=128,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = SignLanguageCNNLSTM(cls.config).to(cls.device)
        
        # Create dummy input
        cls.batch_size = 4
        cls.num_frames = DATA_CONFIG['num_frames']
        cls.frame_size = DATA_CONFIG['frame_size']
        cls.input_shape = (cls.batch_size, cls.num_frames, *cls.frame_size, 3)
    
    def test_model_initialization(self):
        """Test model initialization and architecture."""
        self.assertIsInstance(self.model, SignLanguageCNNLSTM)
        
        # Check CNN layers
        self.assertIsNotNone(self.model.cnn)
        self.assertEqual(self.model.cnn[0].in_channels, 3)
        
        # Check LSTM layers
        self.assertEqual(self.model.lstm.hidden_size, self.config.hidden_size)
        self.assertEqual(self.model.lstm.num_layers, self.config.num_layers)
        self.assertEqual(self.model.lstm.bidirectional, self.config.bidirectional)
        
        # Check output layers
        self.assertEqual(self.model.classifier[-1].out_features, self.config.num_classes)
        self.assertEqual(self.model.bbox_regressor[-1].out_features, 4)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Create input tensor
        x = torch.randn(self.input_shape, device=self.device)
        
        # Forward pass
        class_pred, bbox_pred = self.model(x)
        
        # Check output shapes
        self.assertEqual(
            class_pred.shape,
            (self.batch_size, self.config.num_classes)
        )
        self.assertEqual(
            bbox_pred.shape,
            (self.batch_size, 4)
        )
        
        # Check output values
        self.assertTrue(torch.all(torch.isfinite(class_pred)))
        self.assertTrue(torch.all(torch.isfinite(bbox_pred)))
    
    def test_loss_computation(self):
        """Test loss computation."""
        # Create input tensors
        x = torch.randn(self.input_shape, device=self.device)
        labels = torch.randint(
            0, self.config.num_classes,
            (self.batch_size,),
            device=self.device
        )
        bboxes = torch.rand((self.batch_size, 4), device=self.device)
        
        # Forward pass
        class_pred, bbox_pred = self.model(x)
        
        # Compute losses
        class_loss = torch.nn.CrossEntropyLoss()(class_pred, labels)
        bbox_loss = torch.nn.MSELoss()(bbox_pred, bboxes)
        
        # Check loss values
        self.assertTrue(torch.isfinite(class_loss))
        self.assertTrue(torch.isfinite(bbox_loss))
        self.assertGreater(float(class_loss), 0)
        self.assertGreater(float(bbox_loss), 0)
    
    def test_feature_extraction(self):
        """Test CNN feature extraction."""
        # Create input tensor
        x = torch.randn(self.input_shape, device=self.device)
        
        # Extract features
        features = self.model.extract_features(x)
        
        # Check feature shape
        expected_feature_size = self.model.cnn[-1].out_channels
        self.assertEqual(
            features.shape,
            (self.batch_size, self.num_frames, expected_feature_size)
        )
    
    def test_sequence_processing(self):
        """Test LSTM sequence processing."""
        # Create dummy features
        feature_size = self.model.cnn[-1].out_channels
        features = torch.randn(
            (self.batch_size, self.num_frames, feature_size),
            device=self.device
        )
        
        # Process sequence
        sequence_features = self.model.process_sequence(features)
        
        # Check output shape
        expected_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        self.assertEqual(
            sequence_features.shape,
            (self.batch_size, expected_size)
        )
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Create temporary save directory
        save_dir = Path('test_checkpoints')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / 'test_model.pth'
        
        try:
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__
            }, save_path)
            
            # Create new model
            new_model = SignLanguageCNNLSTM(self.config).to(self.device)
            
            # Load weights
            checkpoint = torch.load(save_path, map_location=self.device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify weights are equal
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
        finally:
            # Cleanup
            if save_path.exists():
                save_path.unlink()
            if save_dir.exists():
                save_dir.rmdir()
    
    def test_cuda_support(self):
        """Test CUDA support if available."""
        if torch.cuda.is_available():
            # Create input tensor
            x = torch.randn(self.input_shape, device=self.device)
            
            # Check model and input are on GPU
            self.assertTrue(next(self.model.parameters()).is_cuda)
            self.assertTrue(x.is_cuda)
            
            # Forward pass
            class_pred, bbox_pred = self.model(x)
            
            # Check outputs are on GPU
            self.assertTrue(class_pred.is_cuda)
            self.assertTrue(bbox_pred.is_cuda)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        # Create input tensors
        x = torch.randn(self.input_shape, device=self.device)
        labels = torch.randint(
            0, self.config.num_classes,
            (self.batch_size,),
            device=self.device
        )
        bboxes = torch.rand((self.batch_size, 4), device=self.device)
        
        # Forward pass
        class_pred, bbox_pred = self.model(x)
        
        # Compute loss
        loss = (
            torch.nn.CrossEntropyLoss()(class_pred, labels) +
            torch.nn.MSELoss()(bbox_pred, bboxes)
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))

def main():
    """Run the tests."""
    unittest.main()

if __name__ == '__main__':
    main()
