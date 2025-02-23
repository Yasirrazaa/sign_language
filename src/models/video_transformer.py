"""Video Transformer model implementation for sign language detection."""

import torch
import torch.nn as nn
import torchvision.models as models
import math
from dataclasses import dataclass

from ..config import MODEL_CONFIG

@dataclass
class TransformerConfig:
    """Configuration for Video Transformer model."""
    num_classes: int
    d_model: int = MODEL_CONFIG['transformer']['d_model']
    nhead: int = MODEL_CONFIG['transformer']['nhead']
    num_encoder_layers: int = MODEL_CONFIG['transformer']['num_encoder_layers']
    dim_feedforward: int = MODEL_CONFIG['transformer']['dim_feedforward']
    dropout_rate: float = MODEL_CONFIG['transformer']['dropout_rate']
    attention_dropout: float = MODEL_CONFIG['transformer']['attention_dropout']
    max_seq_length: int = MODEL_CONFIG['transformer']['max_seq_length']
    activation: str = MODEL_CONFIG['transformer']['activation']

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]

class VideoTransformer(nn.Module):
    """Transformer model for sign language detection."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # CNN backbone (ResNet-50)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]  # Remove final FC layer
        )
        
        # Get CNN output size
        self.feature_size = resnet.fc.in_features
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(
            self.feature_size,
            config.d_model
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.d_model,
            config.max_seq_length
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout_rate,
            activation=config.activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from video frames.
        
        Args:
            x: Input tensor [batch_size, num_frames, channels, height, width]
            
        Returns:
            Features tensor [batch_size, num_frames, d_model]
        """
        batch_size, num_frames = x.shape[:2]
        
        # Reshape for CNN
        x = x.view(-1, *x.shape[2:])  # [batch*frames, C, H, W]
        
        # Extract features
        features = self.backbone(x)  # [batch*frames, features]
        features = features.view(batch_size, num_frames, -1)
        
        # Project to transformer dimension
        features = self.feature_projection(features)
        
        return features

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate attention mask for autoregressive training."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, num_frames, channels, height, width]
            
        Returns:
            Class predictions
        """
        # Extract features
        features = self.extract_features(x)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Generate attention mask
        mask = self.generate_square_subsequent_mask(
            features.size(1),
            features.device
        )
        
        # Transformer encoding
        encoded = self.transformer_encoder(
            features.transpose(0, 1),
            mask
        )
        
        # Use final token for prediction
        final_state = encoded[-1]
        
        # Get predictions
        class_pred = self.classifier(final_state)
        return class_pred
