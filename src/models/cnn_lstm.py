"""CNN-LSTM model implementation for sign language detection."""

import torch
import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass

from ..config import MODEL_CONFIG

@dataclass
class CNNLSTMConfig:
    """Configuration for CNN-LSTM model."""
    num_classes: int
    hidden_size: int = MODEL_CONFIG['cnn_lstm']['hidden_size']
    num_layers: int = MODEL_CONFIG['cnn_lstm']['num_layers']
    dropout_rate: float = MODEL_CONFIG['cnn_lstm']['dropout_rate']
    bidirectional: bool = MODEL_CONFIG['cnn_lstm']['bidirectional']

class SignLanguageCNNLSTM(nn.Module):
    """CNN-LSTM model for sign language detection."""
    
    def __init__(self, config: CNNLSTMConfig):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # CNN backbone (ResNet-50)
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]  # Remove final FC layer
        )
        
        # Get CNN output size
        self.feature_size = resnet.fc.in_features
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Output size calculation
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(lstm_output_size // 2, config.num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from video frames.
        
        Args:
            x: Input tensor [batch_size, num_frames, channels, height, width]
            
        Returns:
            Features tensor [batch_size, num_frames, feature_size]
        """
        batch_size, num_frames = x.shape[:2]
        
        # Reshape for CNN
        x = x.view(-1, *x.shape[2:])  # [batch*frames, C, H, W]
        
        # Extract features
        features = self.cnn(x)  # [batch*frames, features]
        
        # Reshape back
        features = features.view(batch_size, num_frames, -1)
        
        return features
    
    def process_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process feature sequence through LSTM.
        
        Args:
            features: Input features [batch_size, num_frames, feature_size]
            
        Returns:
            LSTM output for final timestep
        """
        # LSTM forward pass
        output, (hidden, _) = self.lstm(features)
        
        if self.config.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return hidden
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, num_frames, channels, height, width]
            
        Returns:
            Class prediction logits
        """
        # Extract CNN features
        features = self.extract_features(x)
        
        # Process through LSTM
        lstm_output = self.process_sequence(features)
        
        # Get predictions
        class_pred = self.classifier(lstm_output)
        return class_pred  # Return raw logits, no softmax