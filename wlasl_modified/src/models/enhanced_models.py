"""Enhanced models for sign language recognition with memory efficiency."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class MemoryEfficientCNNLSTM(nn.Module):
    """Memory-efficient CNN+LSTM implementation."""
    
    def __init__(self,
                 num_classes: int = 26,
                 input_shape: Tuple[int, int, int, int] = (30, 128, 128, 3),
                 conv_filters: Tuple[int] = (64, 128, 256, 512),
                 lstm_units: Tuple[int] = (256, 128),
                 dropout_rate: float = 0.5):
        """
        Initialize enhanced CNN+LSTM model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input shape (frames, height, width, channels)
            conv_filters: Number of filters in Conv2D layers
            lstm_units: Number of units in LSTM layers
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_shape = input_shape
        
        # Time-distributed CNN with shared weights
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, conv_filters[0], kernel_size=3, padding=1),
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1),
            nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1),
            nn.Conv2d(conv_filters[2], conv_filters[3], kernel_size=3, padding=1)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(filters) for filters in conv_filters
        ])
        
        # Calculate CNN output size
        with torch.no_grad():
            x = torch.zeros(1, 3, input_shape[1], input_shape[2])
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                x = F.max_pool2d(F.relu(bn(conv(x))), 2)
            self.cnn_output_size = x.numel()
        
        # LSTM layers with gradient checkpointing
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=self.cnn_output_size if i == 0 else lstm_units[i-1],
                hidden_size=units,
                batch_first=True
            )
            for i, units in enumerate(lstm_units)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layer
        self.classifier = nn.Linear(lstm_units[-1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory optimization.
        
        Args:
            x: Input tensor of shape (batch_size, frames, height, width, channels)
            
        Returns:
            Classification logits
        """
        batch_size, frames = x.size(0), x.size(1)
        
        # Process frames in chunks to save memory
        chunk_size = 8
        cnn_features = []
        
        for i in range(0, frames, chunk_size):
            chunk = x[:, i:i+chunk_size].flatten(0, 1)  # Combine batch and time
            
            # Apply CNN layers with memory efficiency
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                chunk = F.max_pool2d(F.relu(bn(conv(chunk))), 2)
            
            # Reshape features
            chunk = chunk.view(batch_size, -1, self.cnn_output_size)
            cnn_features.append(chunk)
            
            # Clear GPU cache if needed
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Combine chunks
        x = torch.cat(cnn_features, dim=1)
        
        # Apply LSTM layers with gradient checkpointing
        for lstm in self.lstm_layers:
            if self.training:
                def lstm_forward(x): return lstm(x)[0]
                x = torch.utils.checkpoint.checkpoint(lstm_forward, x)
            else:
                x = lstm(x)[0]
            x = self.dropout(x)
        
        # Use last time step for classification
        x = x[:, -1]
        
        # Classification
        x = self.classifier(x)
        
        return x

class MemoryEfficient3DCNN(nn.Module):
    """Memory-efficient 3D CNN implementation."""
    
    def __init__(self,
                 num_classes: int = 26,
                 input_shape: Tuple[int, int, int, int] = (30, 128, 128, 3),
                 base_filters: int = 32):
        """
        Initialize enhanced 3D CNN model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input shape (frames, height, width, channels)
            base_filters: Base number of filters
        """
        super().__init__()
        
        # 3D CNN layers with efficient memory usage
        self.conv3d_layers = nn.ModuleList([
            # Layer 1: temporal resolution preserved
            nn.Conv3d(3, base_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters),
            
            # Layer 2: temporal downsampling
            nn.Conv3d(base_filters, base_filters*2, kernel_size=(3, 3, 3), 
                     stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters*2),
            
            # Layer 3: spatial downsampling
            nn.Conv3d(base_filters*2, base_filters*4, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters*4),
            
            # Layer 4: temporal and spatial downsampling
            nn.Conv3d(base_filters*4, base_filters*8, kernel_size=(3, 3, 3),
                     stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters*8)
        ])
        
        # Calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_shape[:-1])
            for layer in self.conv3d_layers[::2]:  # Skip batch norm layers
                x = F.relu(layer(x))
                x = F.max_pool3d(x, kernel_size=2, stride=2)
            self.conv_output_size = x.numel()
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory optimization.
        
        Args:
            x: Input tensor of shape (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        # Apply 3D CNN layers with intermediate feature cleanup
        for i, layer in enumerate(self.conv3d_layers):
            x = layer(x)
            if i % 2 == 1:  # After each conv-bn pair
                x = F.relu(x)
                x = F.max_pool3d(x, kernel_size=2, stride=2)
                
                # Clear intermediate features
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Global average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def create_memory_efficient_model(
    model_type: str,
    num_classes: int = 26,
    input_shape: Tuple[int, int, int, int] = (30, 128, 128, 3)
) -> nn.Module:
    """
    Create memory-efficient model.
    
    Args:
        model_type: Type of model ('cnn_lstm' or '3dcnn')
        num_classes: Number of output classes
        input_shape: Input shape
        
    Returns:
        Initialized model
    """
    if model_type == 'cnn_lstm':
        return MemoryEfficientCNNLSTM(
            num_classes=num_classes,
            input_shape=input_shape
        )
    elif model_type == '3dcnn':
        return MemoryEfficient3DCNN(
            num_classes=num_classes,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")