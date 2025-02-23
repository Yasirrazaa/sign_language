"""Configuration for hybrid transformer models and their trainers."""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import torch

@dataclass
class CNNTransformerConfig:
    """Configuration for CNN-Transformer model."""
    
    # Model architecture
    num_classes: int = 26
    num_frames: int = 30
    embed_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Memory optimization
    use_checkpoint: bool = True
    chunk_size: int = 128  # Size for chunked attention computation
    mixed_precision: bool = True
    
    # Backbone options
    backbone: str = 'resnet50'
    pretrained: bool = True
    freeze_backbone: bool = False

@dataclass
class TimeSformerConfig:
    """Configuration for TimeSformer model."""
    
    # Model architecture
    num_classes: int = 26
    num_frames: int = 30
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Memory optimization
    use_checkpoint: bool = True
    divided_space_time: bool = True  # Use divided space-time attention
    chunk_size: int = 128
    mixed_precision: bool = True

@dataclass
class TrainerConfig:
    """Configuration for transformer trainers."""
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    gradient_clip: float = 1.0
    
    # Memory optimization
    mixed_precision: bool = True
    cleanup_interval: int = 10
    aggressive_cleanup: bool = True
    pin_memory: bool = True
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Learning rate schedule
    lr_schedule: str = 'cosine'  # 'cosine' or 'reduce_on_plateau'
    min_lr: float = 1e-6
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Checkpointing
    save_frequency: int = 5
    keep_n_checkpoints: int = 3
    
    # Hardware
    num_workers: int = 4
    prefetch_factor: int = 2
    
    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Get optimizer with weight decay handling."""
        # Separate weight decay parameters
        decay = set()
        no_decay = set()
        
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if 'bias' in pn or 'ln' in mn or 'bn' in mn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in model.named_parameters()}
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [param_dict[pn] for pn in sorted(no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return torch.optim.AdamW(optim_groups, lr=self.learning_rate)
    
    def get_scheduler(self, 
                     optimizer: torch.optim.Optimizer
                     ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        if self.lr_schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.min_lr
            )
        elif self.lr_schedule == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_factor,
                patience=self.lr_patience,
                min_lr=self.min_lr
            )
        else:
            return None

@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    data_dir: str = 'data'
    train_list: str = 'train.txt'
    val_list: str = 'val.txt'
    test_list: str = 'test.txt'
    
    # Frame processing
    frame_size: Tuple[int, int] = (224, 224)
    num_frames: int = 30
    temporal_stride: int = 2
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: int = 15
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translation_range: Tuple[float, float] = (-0.2, 0.2)
    
    # Memory optimization
    cache_size: int = 1000  # Number of frames to cache
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True

def get_config(model_name: str) -> Dict:
    """
    Get complete configuration for specified model.
    
    Args:
        model_name: Name of the model ('cnn_transformer' or 'timesformer')
        
    Returns:
        Dictionary containing model, trainer, and data configurations
    """
    if model_name == 'cnn_transformer':
        model_config = CNNTransformerConfig()
    elif model_name == 'timesformer':
        model_config = TimeSformerConfig()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return {
        'model': model_config,
        'trainer': TrainerConfig(),
        'data': DataConfig()
    }

def print_memory_recommendations():
    """Print memory optimization recommendations."""
    recommendations = [
        "Memory Optimization Recommendations:",
        "1. Use gradient checkpointing for large models",
        "2. Enable mixed precision training",
        "3. Use appropriate batch size and gradient accumulation",
        "4. Enable aggressive memory cleanup for TimeSformer",
        "5. Use divided space-time attention for better memory efficiency",
        "6. Monitor memory usage during training",
        "7. Adjust chunk size based on available memory",
        "8. Use appropriate number of workers for data loading",
        "9. Clear cache periodically during training"
    ]
    
    print("\n" + "\n".join(recommendations) + "\n")