"""Memory-efficient I3D-Transformer hybrid architectures for sign language recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from .i3d_base import MemoryEfficientI3D
from .hybrid_transformers import MemoryEfficientAttention, TransformerBlock

class I3DTransformerBase(nn.Module):
    """Base class for I3D-Transformer models."""
    
    def _init_weights(self):
        """Initialize transformer weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.apply(self._init_weights_recursive)
    
    def _init_weights_recursive(self, m):
        """Initialize weights recursively."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _clear_memory(self):
        """Clear unused memory."""
        if hasattr(self, 'last_hidden'):
            del self.last_hidden
        torch.cuda.empty_cache()

class I3DTransformer(I3DTransformerBase):
    """Memory-efficient I3D with unified space-time attention."""
    
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_checkpoint: bool = True,
        i3d_pretrained_path: Optional[str] = None
    ):
        """Initialize I3D-Transformer model."""
        super().__init__()
        
        # I3D backbone with memory efficiency
        self.i3d = MemoryEfficientI3D(
            num_classes=num_classes,
            spatial_squeeze=True,
            final_endpoint='Logits',
            use_checkpointing=use_checkpoint
        )
        if i3d_pretrained_path:
            self.i3d.load_state_dict(torch.load(i3d_pretrained_path))
        
        # Projection from I3D features
        self.projection = nn.Linear(1024, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient computation."""
        B = x.shape[0]
        
        # Extract I3D features with checkpointing
        features = self.i3d.extract_features(x)
        
        # Process features
        features = features.transpose(1, 2)  # [B, C, T, H, W] -> [B, T, C, H, W]
        features = features.mean([-2, -1])   # Pool spatial dimensions
        
        # Project to embedding dimension
        x = self.projection(features)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Apply transformer blocks with checkpointing
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        self._clear_memory()
        return x

class I3DTimeSformer(I3DTransformerBase):
    """Memory-efficient I3D with divided space-time attention."""
    
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_checkpoint: bool = True,
        i3d_pretrained_path: Optional[str] = None
    ):
        """Initialize I3D-TimeSformer model."""
        super().__init__()
        
        # I3D backbone with memory efficiency
        self.i3d = MemoryEfficientI3D(
            num_classes=num_classes,
            spatial_squeeze=True,
            final_endpoint='Logits',
            use_checkpointing=use_checkpoint
        )
        if i3d_pretrained_path:
            self.i3d.load_state_dict(torch.load(i3d_pretrained_path))
        
        # Projection
        self.projection = nn.Linear(1024, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer blocks (alternating temporal and spatial attention)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            # Temporal attention
            self.blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                )
            )
            # Spatial attention
            self.blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                )
            )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with divided space-time attention."""
        # Extract I3D features
        features = self.i3d.extract_features(x)
        
        # Process features
        features = features.transpose(1, 2)  # [B, C, T, H, W] -> [B, T, C, H, W]
        features = features.mean([-2, -1])   # Pool spatial dimensions
        
        # Project to embedding dimension
        x = self.projection(features)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Process with alternating temporal and spatial attention
        for i in range(0, len(self.blocks), 2):
            # Temporal attention
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x + self.temporal_embed[:, :x.size(1)]
                )
            else:
                x = self.blocks[i](x + self.temporal_embed[:, :x.size(1)])
            
            # Spatial attention
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(self.blocks[i+1], x)
            else:
                x = self.blocks[i+1](x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        self._clear_memory()
        return x

def create_model(
    model_name: str,
    num_classes: int,
    num_frames: int,
    **kwargs
) -> nn.Module:
    """
    Create an I3D-based model instance.
    
    Args:
        model_name: Name of the model ('i3d_transformer' or 'i3d_timesformer')
        num_classes: Number of output classes
        num_frames: Number of input frames
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    if model_name == 'i3d_transformer':
        return I3DTransformer(
            num_classes=num_classes,
            num_frames=num_frames,
            **kwargs
        )
    elif model_name == 'i3d_timesformer':
        return I3DTimeSformer(
            num_classes=num_classes,
            num_frames=num_frames,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")