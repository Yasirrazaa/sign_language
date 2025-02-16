"""Test script for Video Transformer model implementation."""

import unittest
import torch
import numpy as np
from pathlib import Path

from models.video_transformer import VideoTransformer, TransformerConfig
from src.config import MODEL_CONFIG, DATA_CONFIG

class TestVideoTransformer(unittest.TestCase):
    """Test cases for VideoTransformer model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = TransformerConfig(
            num_classes=10,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = VideoTransformer(cls.config).to(cls.device)
        
        # Create dummy input
        cls.batch_size = 4
        cls.num_frames = DATA_CONFIG['num_frames']
        cls.frame_size = DATA_CONFIG['frame_size']
        cls.input_shape = (cls.batch_size, cls.num_frames, *cls.frame_size, 3)
    
    def test_model_initialization(self):
        """Test model initialization and architecture."""
        self.assertIsInstance(self.model, VideoTransformer)
        
        # Check backbone CNN
        self.assertIsNotNone(self.model.backbone)
        self.assertEqual(self.model.backbone[0].in_channels, 3)
        
        # Check transformer encoder
        self.assertEqual(
            self.model.transformer_encoder.layers[0].self_attn.num_heads,
            self.config.nhead
        )
        self.assertEqual(
            self.model.transformer_encoder.layers[0].linear1.out_features,
            self.config.dim_feedforward
        )
        
        # Check output layers
        self.assertEqual(
            self.model.classifier[-1].out_features,
            self.config.num_classes
        )
        self.assertEqual(self.model.bbox_head[-1].out_features, 4)
    
    def test_positional_encoding(self):
        """Test positional encoding generation."""
        # Generate positional encoding
        pos_encoding = self.model.pos_encoder.pe
        
        # Check shape and properties
        self.assertEqual(
            pos_encoding.shape,
            (1, self.config.max_seq_length, self.config.d_model)
        )
        self.assertTrue(torch.all(torch.isfinite(pos_encoding)))
        
        # Test different positions encode differently
        self.assertFalse(torch.allclose(
            pos_encoding[0, 0],
            pos_encoding[0, 1]
        ))
    
    def test_attention_mechanism(self):
        """Test self-attention mechanism."""
        # Create input features
        features = torch.randn(
            self.batch_size,
            self.num_frames,
            self.config.d_model,
            device=self.device
        )
        
        # Get attention weights from first layer
        with torch.no_grad():
            # Forward pass through transformer
            output = self.model.transformer_encoder(features)
            
            # Get attention weights from first layer
            attn_weights = self.model.transformer_encoder.layers[0].self_attn._get_attention_weights(
                features, features, features
            )
        
        # Check attention properties
        self.assertEqual(
            attn_weights.shape,
            (self.batch_size * self.config.nhead, self.num_frames, self.num_frames)
        )
        self.assertTrue(torch.all(torch.isfinite(attn_weights)))
        self.assertTrue(torch.all(attn_weights >= 0))  # After softmax
        
        # Check attention sums to 1 along correct dimension
        attn_sums = torch.sum(attn_weights, dim=-1)
        self.assertTrue(torch.allclose(attn_sums, torch.ones_like(attn_sums)))
    
    def test_feature_extraction(self):
        """Test CNN feature extraction."""
        # Create input tensor
        x = torch.randn(self.input_shape, device=self.device)
        
        # Extract features
        features = self.model.extract_features(x)
        
        # Check feature shape
        self.assertEqual(
            features.shape,
            (self.batch_size, self.num_frames, self.config.d_model)
        )
        self.assertTrue(torch.all(torch.isfinite(features)))
    
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
        self.assertEqual(bbox_pred.shape, (self.batch_size, 4))
        
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
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Create temporary save directory
        save_dir = Path('test_checkpoints')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / 'test_transformer.pth'
        
        try:
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__
            }, save_path)
            
            # Create new model
            new_model = VideoTransformer(self.config).to(self.device)
            
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
        
        # Zero gradients
        self.model.zero_grad()
        
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
    
    def test_sequence_masking(self):
        """Test attention masking mechanism."""
        # Create sequence mask
        seq_length = 10
        mask = self.model.generate_square_subsequent_mask(seq_length)
        
        # Check mask shape and properties
        self.assertEqual(mask.shape, (seq_length, seq_length))
        self.assertTrue(torch.all(torch.isfinite(mask)))
        
        # Check masking pattern
        for i in range(seq_length):
            for j in range(seq_length):
                if i < j:
                    self.assertEqual(mask[i, j], float('-inf'))
                else:
                    self.assertEqual(mask[i, j], 0)

def main():
    """Run the tests."""
    unittest.main()

if __name__ == '__main__':
    main()
