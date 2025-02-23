"""Memory-efficient sign language recognition model with temporal modeling and attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class TemporalShiftModule(nn.Module):
    """Memory-efficient temporal shift module."""
    
    def __init__(self, net: nn.Module, n_segment: int = 8, n_div: int = 8):
        """
        Initialize temporal shift module.
        
        Args:
            net: Base network module
            n_segment: Number of frames
            n_div: Number of divided channels for shifting
        """
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient shifting."""
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)
    
    @staticmethod
    def shift(x: torch.Tensor, n_segment: int, fold_div: int = 8) -> torch.Tensor:
        """Perform temporal shift operation efficiently."""
        B, C, T, H, W = x.size()
        x = x.view(B, C, n_segment, T // n_segment, H, W)
        
        # Memory-efficient implementation: only shift necessary channels
        fold = C // fold_div
        out = torch.zeros_like(x)
        
        # Shift left
        out[:, :fold] = x[:, :fold, 1:]
        
        # Shift right
        out[:, fold:2*fold] = x[:, fold:2*fold, :-1]
        
        # No shift
        out[:, 2*fold:] = x[:, 2*fold:]
        
        return out.view(B, C, T, H, W)

class TemporalPyramidPooling(nn.Module):
    """Memory-efficient temporal pyramid pooling."""
    
    def __init__(self, levels: Tuple[int] = (1, 2, 3, 6)):
        """
        Initialize temporal pyramid pooling.
        
        Args:
            levels: Pooling levels for temporal pyramid
        """
        super().__init__()
        self.levels = levels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive pooling."""
        B, C, T, H, W = x.size()
        out = []
        
        for level in self.levels:
            # Use adaptive pooling for memory efficiency
            tensor = F.adaptive_avg_pool3d(x, output_size=(level, 1, 1))
            out.append(tensor.view(B, C, level, 1, 1))
        
        # Efficient concatenation
        out = torch.cat([F.interpolate(t, size=(T, H, W)) for t in out], dim=1)
        return out

class SignAttention(nn.Module):
    """Memory-efficient sign language specific attention."""
    
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 value_channels: int,
                 num_heads: int = 8):
        """
        Initialize sign attention module.
        
        Args:
            in_channels: Input channel size
            key_channels: Key dimension
            value_channels: Value dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        
        # Efficient 1x1 convolutions for attention
        self.keys = nn.Conv3d(in_channels, key_channels, 1, bias=False)
        self.queries = nn.Conv3d(in_channels, key_channels, 1, bias=False)
        self.values = nn.Conv3d(in_channels, value_channels, 1, bias=False)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient attention computation."""
        B, C, T, H, W = x.size()
        
        # Compute keys, queries, and values efficiently
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)
        
        # Reshape for multi-head attention
        keys = keys.view(B, self.num_heads, -1, T*H*W)
        queries = queries.view(B, self.num_heads, -1, T*H*W)
        values = values.view(B, self.num_heads, -1, T*H*W)
        
        # Efficient attention computation with chunking
        head_dim = keys.size(2)
        chunk_size = 128  # Adjust based on available memory
        
        attention_weights = []
        for i in range(0, T*H*W, chunk_size):
            # Process attention in chunks
            q_chunk = queries[..., i:i+chunk_size]
            chunk_weights = torch.matmul(keys, q_chunk)
            chunk_weights = chunk_weights / math.sqrt(head_dim)
            attention_weights.append(F.softmax(chunk_weights, dim=-1))
        
        # Combine chunks
        attention = torch.cat(attention_weights, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(values, attention)
        out = out.view(B, -1, T, H, W)
        
        # Final projection
        out = self.reprojection(out)
        
        return out

class EfficientSignNet(nn.Module):
    """Memory-efficient sign language recognition network."""
    
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_frames: int = 16):
        """
        Initialize efficient sign language network.
        
        Args:
            num_classes: Number of sign classes
            in_channels: Input channel size
            base_channels: Base channel size
            num_frames: Number of input frames
        """
        super().__init__()
        
        # Base convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Temporal shift module
        self.temporal_shift = TemporalShiftModule(
            nn.Sequential(
                nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True)
            ),
            n_segment=num_frames
        )
        
        # Temporal pyramid pooling
        self.temporal_pyramid = TemporalPyramidPooling()
        
        # Sign attention
        self.sign_attention = SignAttention(
            in_channels=base_channels*2,
            key_channels=base_channels,
            value_channels=base_channels*2
        )
        
        # Classification head
        num_features = base_channels * 2 * len(self.temporal_pyramid.levels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights efficiently."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory-efficient computation.
        
        Args:
            x: Input tensor of shape (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        # Base features
        x = self.conv1(x)
        
        # Apply temporal shift
        x = self.temporal_shift(x)
        
        # Apply sign attention
        x = self.sign_attention(x)
        
        # Apply temporal pyramid pooling
        x = self.temporal_pyramid(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    @torch.jit.ignore
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for visualization or analysis.
        Not included in TorchScript for efficiency.
        """
        x = self.conv1(x)
        x = self.temporal_shift(x)
        x = self.sign_attention(x)
        return x

def create_efficient_sign_net(
    num_classes: int,
    in_channels: int = 3,
    base_channels: int = 64,
    num_frames: int = 16
) -> EfficientSignNet:
    """
    Create memory-efficient sign language recognition model.
    
    Args:
        num_classes: Number of sign classes
        in_channels: Input channel size
        base_channels: Base channel size
        num_frames: Number of input frames
        
    Returns:
        Initialized EfficientSignNet model
    """
    model = EfficientSignNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        num_frames=num_frames
    )
    return model

if __name__ == '__main__':
    # Test model
    model = create_efficient_sign_net(num_classes=100)
    x = torch.randn(2, 3, 16, 224, 224)  # (batch, channels, frames, height, width)
    output = model(x)
    print(f"Output shape: {output.shape}")