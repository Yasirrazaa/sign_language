"""Configuration for memory-efficient sign language recognition models."""

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class EfficientSignNetConfig:
    """Configuration for EfficientSignNet model."""
    
    # Model architecture
    num_classes: int = 100
    in_channels: int = 3
    base_channels: int = 64
    num_frames: int = 16
    
    # Temporal Shift Module
    tsm_segments: int = 8
    tsm_fold_div: int = 8
    
    # Temporal Pyramid Pooling
    tpp_levels: Tuple[int] = (1, 2, 3, 6)
    
    # Sign Attention
    attention_heads: int = 8
    key_channels: int = 64
    value_channels: int = 128
    attention_chunk_size: int = 128
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    optimize_attention_memory: bool = True
    enable_feature_caching: bool = False
    cache_size_limit_mb: int = 1024  # 1GB limit for feature caching

@dataclass
class MemoryOptimizedTrainingConfig:
    """Configuration for memory-optimized training."""
    
    # Basic training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Memory optimization
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    empty_cache_freq: int = 10  # Empty CUDA cache every N steps
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    
    # Memory monitoring
    memory_warning_threshold: float = 0.9  # 90% memory usage warning
    memory_critical_threshold: float = 0.95  # 95% memory usage critical
    enable_memory_tracking: bool = True
    track_memory_every_n_steps: int = 50
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Learning rate scheduling
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

@dataclass
class DataConfig:
    """Configuration for data processing and augmentation."""
    
    # Frame processing
    frame_size: Tuple[int, int] = (224, 224)
    temporal_crop_size: int = 16
    temporal_stride: int = 2
    
    # Augmentation
    enable_augmentation: bool = True
    rotation_range: int = 15
    scale_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip_prob: float = 0.5
    temporal_mask_prob: float = 0.3
    
    # Memory optimization
    chunk_size: int = 32
    max_frames_in_memory: int = 256
    enable_frame_caching: bool = True
    cache_size_limit_mb: int = 2048  # 2GB limit for frame caching

def get_default_configs():
    """Get default configurations."""
    return {
        'model': EfficientSignNetConfig(),
        'training': MemoryOptimizedTrainingConfig(),
        'data': DataConfig()
    }

def validate_memory_settings(configs: dict) -> List[str]:
    """
    Validate memory-related settings and provide warnings.
    
    Args:
        configs: Dictionary of configurations
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check batch size and gradient accumulation
    effective_batch = (configs['training'].batch_size * 
                      configs['training'].gradient_accumulation_steps)
    if effective_batch > 64:
        warnings.append(
            f"Effective batch size {effective_batch} might require too much memory."
            " Consider reducing batch_size or increasing gradient_accumulation_steps."
        )
    
    # Check feature caching
    if (configs['model'].enable_feature_caching and 
        configs['data'].enable_frame_caching):
        total_cache = (configs['model'].cache_size_limit_mb + 
                      configs['data'].cache_size_limit_mb)
        if total_cache > 4096:  # 4GB
            warnings.append(
                f"Total cache size {total_cache}MB might be too large."
                " Consider reducing cache_size_limit_mb."
            )
    
    # Check worker settings
    if configs['training'].num_workers * configs['training'].prefetch_factor > 16:
        warnings.append(
            "High number of workers and prefetch factor might consume too much memory."
            " Consider reducing num_workers or prefetch_factor."
        )
    
    return warnings

def print_memory_optimization_tips():
    """Print tips for memory optimization."""
    tips = [
        "1. Use gradient accumulation for larger effective batch sizes",
        "2. Enable mixed precision training for reduced memory usage",
        "3. Use gradient checkpointing for very deep models",
        "4. Monitor and adjust cache sizes based on available memory",
        "5. Clear unused features and CUDA cache regularly",
        "6. Use streaming dataloader for large datasets",
        "7. Adjust chunk sizes based on GPU memory",
        "8. Enable temporal striding for long sequences"
    ]
    
    print("\nMemory Optimization Tips:")
    for tip in tips:
        print(tip)