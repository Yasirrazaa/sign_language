"""Configuration for memory-efficient sign language recognition."""

from pathlib import Path

# Directory structure
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'processed'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
LOG_DIR = BASE_DIR / 'logs'

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'frame_size': (224, 224),    # Target frame dimensions
    'target_fps': 25,            # Target frames per second
    'chunk_size': 32,            # Number of frames to process at once
    'max_frames': 64,            # Maximum frames per video
    'num_workers': 4,            # Number of preprocessing workers
    'compression_quality': 95,    # JPEG compression quality
    'tmp_cleanup_interval': 100,  # Clean temporary files every N videos
}

# Data loading configuration
DATA_CONFIG = {
    'batch_size': 16,            # Smaller batch size for memory efficiency
    'num_workers': 4,            # Number of data loading workers
    'frame_cache_size': 1000,    # Number of frames to cache in memory
    'pin_memory': True,          # Pin memory for faster GPU transfer
    'prefetch_factor': 2,        # Number of batches to prefetch
    'persistent_workers': True,   # Keep workers alive between epochs
}

# Training configuration
TRAIN_CONFIG = {
    'use_mixed_precision': True,  # Enable automatic mixed precision
    'gradient_accumulation_steps': 4,  # Accumulate gradients for larger effective batch
    'gradient_clip': 1.0,        # Gradient clipping threshold
    
    # Memory optimization
    'aggressive_memory_cleanup': True,  # Enable aggressive memory cleanup
    'checkpoint_frequency': 5,    # Save checkpoints every N epochs
    'keep_n_checkpoints': 3,     # Number of checkpoints to keep
    
    # Training parameters
    'num_epochs': 150,
    'early_stopping_patience': 15,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'lr_scheduler': {
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6
    },
    
    # Validation
    'val_frequency': 1,          # Validate every N epochs
}

# Model configurations
I3D_CONFIG = {
    'in_channels': 3,
    'num_classes': None,  # Set dynamically
    'dropout_prob': 0.5,
    'spatial_squeeze': True,
    'final_endpoint': 'Mixed_5c',
    # Memory efficient settings
    'use_checkpointing': True,   # Use gradient checkpointing
    'efficient_attention': True,  # Use memory-efficient attention
}

TGCN_CONFIG = {
    'input_size': 150,  # 25 joints × 3 coordinates
    'hidden_size': 512,
    'num_classes': None,  # Set dynamically
    'num_layers': 2,
    'dropout': 0.5,
    # Memory efficient settings
    'use_checkpointing': True,
    'bidirectional': True,
    'batch_first': True
}

# Evaluation configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'batch_size': 8,             # Smaller batch size for evaluation
    'num_workers': 2,            # Fewer workers for evaluation
    'save_predictions': True,     # Save model predictions
    'confusion_matrix': True     # Generate confusion matrix
}

# Logging configuration
LOG_CONFIG = {
    'log_frequency': 100,        # Log every N batches
    'log_memory_usage': True,    # Track memory usage
    'save_memory_stats': True,   # Save memory statistics
    'log_gradients': False,      # Disable gradient logging to save memory
}

# Memory monitoring thresholds (in GB)
MEMORY_THRESHOLDS = {
    'gpu_warning': 0.9,          # GPU memory warning threshold (90%)
    'gpu_critical': 0.95,        # GPU memory critical threshold (95%)
    'cpu_warning': 0.8,          # CPU memory warning threshold (80%)
    'cpu_critical': 0.9          # CPU memory critical threshold (90%)
}

# Create required directories
for directory in [PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
