"""Base configuration settings for the sign language recognition project."""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_VIDEOS_DIR = DATA_DIR / 'raw_videos'
PROCESSED_VIDEOS_DIR = DATA_DIR / 'processed_videos'
FRAMES_DIR = DATA_DIR / 'frames'
FEATURES_DIR = DATA_DIR / 'features'
CHECKPOINTS_DIR = BASE_DIR / 'models' / 'checkpoints'
WEIGHTS_DIR = BASE_DIR / 'models' / 'weights'
LOG_DIR = BASE_DIR / 'logs'
ANALYSIS_DIR = BASE_DIR / 'analysis'

# Ensure directories exist
for dir_path in [RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, FRAMES_DIR, FEATURES_DIR,
                 CHECKPOINTS_DIR, WEIGHTS_DIR, LOG_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'frame_size': (224, 224),  # I3D default input size
    'num_frames': 64,          # Temporal coverage
    'fps': 25,                # Standard frame rate
    'train_split': 0.7,       # Training data split
    'val_split': 0.15,        # Validation data split
    'test_split': 0.15,       # Testing data split
    'min_videos_per_class': 20,  # Minimum videos required per class
    
    # Memory-efficient processing
    'batch_size': 8,          # Smaller batch size for memory efficiency
    'num_workers': 4,         # Parallel data loading
    'pin_memory': True,       # Faster data transfer to GPU
    'prefetch_factor': 2,     # Prefetch batches
    
    # Data augmentation
    'brightness_delta': 0.2,    
    'contrast_range': (0.8, 1.2),
    'rotation_range': 15,     
    'zoom_range': 0.1,
    'random_crop': True,
    'random_flip': True
}

# Model configurations
I3D_CONFIG = {
    'in_channels': 3,
    'num_classes': None,  # Set dynamically
    'dropout_prob': 0.5,
    'init_lr': 0.01,
    'weight_decay': 1e-7,
    'momentum': 0.9
}

TGCN_CONFIG = {
    'input_size': 150,  # 25 joints × 3 coordinates
    'hidden_size': 512,
    'num_classes': None,  # Set dynamically
    'num_layers': 2,
    'dropout': 0.5
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 150,
    'num_folds': 5,            # Cross-validation folds
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-7,
    'warmup_epochs': 5,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 8,
    'reduce_lr_factor': 0.3,
    'mixed_precision': True    # Memory efficient training
}

# Evaluation metrics
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'top_k': [1, 3, 5],
    'confusion_matrix': True,
    'save_predictions': True,
    'visualization_samples': 10
}

# Preprocessing memory optimization
PREPROCESSING_CONFIG = {
    'chunk_size': 32,          # Process videos in chunks
    'max_memory_usage': '80%', # Maximum memory usage
    'tmp_dir': DATA_DIR / 'tmp',
    'cleanup_tmp': True,
    'compression': 'JPEG',     # Frame compression format
    'compression_quality': 95  # JPEG quality for frame saving
}