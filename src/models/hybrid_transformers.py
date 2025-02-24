"""Memory-efficient hybrid transformer architectures for sign language recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Tuple
import math

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient implementation of multi-head attention."""
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attention_dropout: float = 0.0):
        """
        Initialize attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Use bias in QKV projection
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with chunked attention computation."""
        B, N, C = x.shape
        
        # Compute QKV with chunk processing
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                       three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores in chunks
        chunk_size = 128  # Adjust based on available memory
        num_chunks = (N + chunk_size - 1) // chunk_size
        
        out = torch.zeros_like(q)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            
            # Compute chunk attention scores
            chunk_scores = torch.matmul(q[:, :, start_idx:end_idx],
                                      k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                chunk_scores = chunk_scores.masked_fill(
                    mask[:, start_idx:end_idx] == 0, float('-inf'))
            
            chunk_attn = F.softmax(chunk_scores, dim=-1)
            chunk_attn = self.dropout(chunk_attn)
            
            # Update output
            out[:, :, start_idx:end_idx] = torch.matmul(chunk_attn, v)
        
        # Final projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """Memory-efficient transformer block."""
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0):
        """Initialize transformer block."""
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemoryEfficientAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformer(nn.Module):
    """Hybrid CNN-Transformer model with memory optimizations."""
    
    def __init__(self,
                 num_classes: int,
                 num_frames: int,
                 embed_dim: int = 512,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 use_checkpoint: bool = True):
        """
        Initialize CNN-Transformer model.
        
        Args:
            num_classes: Number of output classes
            num_frames: Number of input frames
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Use bias in QKV projection
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        
        # CNN backbone (ResNet-50 pretrained)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0',
                                     'resnet50', pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Temporal and spatial projection
        self.projection = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
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
        if use_checkpoint:
            self.backbone.requires_grad_(True)
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory-efficient computation.
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Classification logits
        """
        B, T = x.shape[:2]
        
        # Process frames through CNN backbone
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.backbone, x)
        else:
            x = self.backbone(x)
        
        # Project features
        x = self.projection(x)
        
        # Pool spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1).squeeze(-1)
        
        # Reshape sequence
        x = rearrange(x, '(b t) d -> b t d', b=B)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        # Global temporal pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        return x

class TimeSformer(nn.Module):
    """Memory-efficient TimeSformer implementation."""
    
    def __init__(self,
                 num_classes: int,
                 num_frames: int,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 use_checkpoint: bool = True):
        """
        Initialize TimeSformer model.
        
        Args:
            num_classes: Number of output classes
            num_frames: Number of input frames
            img_size: Input image size
            patch_size: Size of image patches
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Use bias in QKV projection
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, 
                                   kernel_size=patch_size, 
                                   stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
        # Transformer blocks
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
    
    def _init_weights(self):
        """Initialize transformer weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with divided space-time attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Classification logits
        """
        B, T, C, H, W = x.shape
        
        # Patch embedding
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = rearrange(x, 'bt d h w -> bt (h w) d')
        
        # Add spatial positional embedding
        x = x + self.pos_embed
        
        # Reshape for temporal processing
        x = rearrange(x, '(b t) n d -> b t n d', b=B)
        
        # Process with transformer blocks
        for i in range(0, len(self.blocks), 2):
            # Temporal attention
            xt = rearrange(x, 'b t n d -> (b n) t d')
            if self.use_checkpoint and self.training:
                xt = torch.utils.checkpoint.checkpoint(self.blocks[i], xt)
            else:
                xt = self.blocks[i](xt)
            xt = rearrange(xt, '(b n) t d -> b t n d', b=B)
            
            # Spatial attention
            xs = rearrange(xt, 'b t n d -> (b t) n d')
            if self.use_checkpoint and self.training:
                xs = torch.utils.checkpoint.checkpoint(self.blocks[i+1], xs)
            else:
                xs = self.blocks[i+1](xs)
            x = rearrange(xs, '(b t) n d -> b t n d', b=B)
        
        # Global pooling
        x = x.mean(dim=[1, 2])
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        return x

def create_model(
    model_name: str,
    num_classes: int,
    num_frames: int,
    **kwargs
) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model ('cnn_transformer' or 'timesformer')
        num_classes: Number of output classes
        num_frames: Number of input frames
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    if model_name == 'cnn_transformer':
        return CNNTransformer(
            num_classes=num_classes,
            num_frames=num_frames,
            **kwargs
        )
    elif model_name == 'timesformer':
        return TimeSformer(
            num_classes=num_classes,
            num_frames=num_frames,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")