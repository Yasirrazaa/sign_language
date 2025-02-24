"""Memory-efficient I3D implementation based on the original Inception I3D architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

class MemoryEfficientMaxPool3d(nn.MaxPool3d):
    """Memory-efficient 3D max pooling with 'same' padding."""
    
    def compute_pad(self, dim: int, s: int) -> int:
        """Compute padding size."""
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic padding."""
        batch, channel, t, h, w = x.size()
        
        # Calculate output dimensions
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        
        # Calculate padding
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        
        # Apply padding
        pad = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2
        )
        x = F.pad(x, pad)
        
        return super().forward(x)

class MemoryEfficientUnit3D(nn.Module):
    """Memory-efficient 3D convolution unit."""
    
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape: Tuple[int, int, int] = (1, 1, 1),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: int = 0,
        activation_fn: Optional[nn.Module] = F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = 'unit_3d'
    ):
        """Initialize Unit3D module."""
        super().__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        # Initialize conv3d with efficient memory layout
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # Dynamic padding in forward
            bias=self._use_bias
        )
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(
                self._output_channels,
                eps=0.001,
                momentum=0.01
            )

    def compute_pad(self, dim: int, s: int) -> int:
        """Compute padding size."""
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic padding."""
        batch, channel, t, h, w = x.size()
        
        # Calculate padding
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        
        # Apply padding efficiently
        pad = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_t // 2, pad_t - pad_t // 2
        )
        x = F.pad(x, pad)
        
        # Convolution
        x = self.conv3d(x)
        
        # Batch norm and activation
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        
        return x

class MemoryEfficientInceptionModule(nn.Module):
    """Memory-efficient Inception module."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        name: str
    ):
        """Initialize Inception module."""
        super().__init__()
        
        self.b0 = MemoryEfficientUnit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=f'{name}/Branch_0/Conv3d_0a_1x1'
        )
        self.b1a = MemoryEfficientUnit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=f'{name}/Branch_1/Conv3d_0a_1x1'
        )
        self.b1b = MemoryEfficientUnit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=f'{name}/Branch_1/Conv3d_0b_3x3'
        )
        self.b2a = MemoryEfficientUnit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=f'{name}/Branch_2/Conv3d_0a_1x1'
        )
        self.b2b = MemoryEfficientUnit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=f'{name}/Branch_2/Conv3d_0b_3x3'
        )
        self.b3a = MemoryEfficientMaxPool3d(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding=0
        )
        self.b3b = MemoryEfficientUnit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=f'{name}/Branch_3/Conv3d_0b_1x1'
        )
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with efficient memory usage."""
        # Process branches separately to save memory
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        
        # Concatenate results
        return torch.cat([b0, b1, b2, b3], dim=1)

class MemoryEfficientI3D(nn.Module):
    """Memory-efficient I3D implementation."""
    
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions'
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        final_endpoint: str = 'Logits',
        name: str = 'inception_i3d',
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
        use_checkpointing: bool = False
    ):
        """Initialize I3D model."""
        super().__init__()
        
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown final endpoint {final_endpoint}')

        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.use_checkpointing = use_checkpointing
        
        self._build_model(in_channels, name, dropout_keep_prob)

    def _build_model(
        self,
        in_channels: int,
        name: str,
        dropout_keep_prob: float
    ):
        """Build I3D architecture."""
        self.end_points = {}
        
        def add_sequential_module(endpoints: List[str]) -> None:
            """Add modules sequentially with optional checkpointing."""
            for end_point in endpoints:
                if hasattr(self, f'_build_{end_point}'):
                    module = getattr(self, f'_build_{end_point}')(
                        in_channels if end_point == 'Conv3d_1a_7x7' else None,
                        name
                    )
                    self.end_points[end_point] = module
                    self.add_module(end_point, module)
                    
                    if self._final_endpoint == end_point:
                        break

        # Build network sequentially
        add_sequential_module(self.VALID_ENDPOINTS)
        
        # Add logits layer
        if self._final_endpoint == 'Logits':
            self._build_logits_layer(dropout_keep_prob)
        
        # Initialize the network
        self._initialize_weights()

    def _build_logits_layer(self, dropout_keep_prob: float):
        """Build logits layer."""
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = MemoryEfficientUnit3D(
            in_channels=384+384+128+128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits'
        )

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        pretrained: bool = False,
        n_tune_layers: int = -1
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if pretrained:
            assert n_tune_layers >= 0
            freeze_endpoints = self.VALID_ENDPOINTS[:-n_tune_layers]
            tune_endpoints = self.VALID_ENDPOINTS[-n_tune_layers:]
        else:
            freeze_endpoints = []
            tune_endpoints = self.VALID_ENDPOINTS

        # Process frozen layers
        with torch.no_grad():
            for end_point in freeze_endpoints:
                if end_point in self.end_points:
                    module = self._modules[end_point]
                    x = module(x)

        # Process tunable layers
        for end_point in tune_endpoints:
            if end_point in self.end_points:
                module = self._modules[end_point]
                if self.use_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(module, x)
                else:
                    x = module(x)

        # Apply final layers
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        for end_point in self.VALID_ENDPOINTS[:-1]:  # Exclude 'Logits' and 'Predictions'
            if end_point in self.end_points:
                if self.use_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(self._modules[end_point], x)
                else:
                    x = self._modules[end_point](x)
        return self.avg_pool(x)

    def replace_logits(self, num_classes: int):
        """Replace logits layer for fine-tuning."""
        self._num_classes = num_classes
        self.logits = MemoryEfficientUnit3D(
            in_channels=384+384+128+128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits'
        )