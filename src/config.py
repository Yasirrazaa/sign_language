"""Configuration settings for the sign language detection project."""

from pathlib import Path

# Directory paths
BASE_DIR = Path('/media/yasir/D/sign language')
VIDEO_DIR = BASE_DIR / 'video'
PROCESSED_DIR = BASE_DIR / 'processed'
LOG_DIR = BASE_DIR / 'logs'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

# Data configuration
DATA_CONFIG = {
    'frame_size': (192, 192),  # Balanced size for memory and detail
    'num_frames': 48,          # Increased frames for better temporal coverage
    'fps': 25,                # Target frames per second
    'train_split': 0.7,       # Training data split
    'val_split': 0.15,        # Validation data split
    'test_split': 0.15,       # Testing data split
    'min_videos_per_class': 20,  # Minimum videos required per class
    
    # Data augmentation parameters
    'brightness_delta': 0.2,    # Random brightness adjustment range
    'contrast_range': (0.8, 1.2),  # Random contrast adjustment range
    'rotation_range': 15,     # Maximum rotation angle in degrees
    'zoom_range': 0.1,        # Maximum zoom range
    'batch_size': 12          # Balanced batch size
}

# MediaPipe hand detection configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,  # Use video tracking optimization
    'max_num_hands': 2,          # Maximum number of hands to detect
    'min_detection_confidence': 0.7,  # Minimum confidence for detection
    'min_tracking_confidence': 0.5    # Minimum confidence for tracking
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': DATA_CONFIG['batch_size'],
    'epochs': 100,               # Maximum number of training epochs
    'num_folds': 7,             # Number of cross-validation folds
    'initial_learning_rate': 5e-5,  # Reduced initial learning rate for stability
    'min_learning_rate': 1e-7,     # Minimum learning rate for reduction
    'early_stopping_patience': 15,  # Increased patience for better convergence
    'reduce_lr_patience': 7,       # Increased patience for learning rate reduction
    'random_seed': 42            # Random seed for reproducibility
}

# Model configuration
MODEL_CONFIG = {
    # CNN+LSTM model configuration
    'cnn_lstm': {
        'hidden_size': 384,      # Increased for better capacity
        'num_layers': 3,         # Keep current
        'dropout_rate': 0.5,     # Strong regularization
        'bidirectional': True,   # Using bidirectional LSTM
        'lstm_units': [384, 192],  # Balanced unit sizes
        'dense_units': [384],    # Increased dense layer
        'l2_reg': 1e-5          # L2 regularization factor
    },
    
    # Transformer model configuration
    'transformer': {
        'd_model': 192,         # Increased model dimension
        'nhead': 8,             # Number of attention heads
        'num_encoder_layers': 6, # Keep current
        'dim_feedforward': 768,  # Increased feedforward dimension
        'dropout_rate': 0.3,     # Moderate dropout
        'attention_dropout': 0.2, # Attention-specific dropout
        'max_seq_length': 1000,  # Maximum sequence length
        'activation': 'gelu'     # Using GELU activation
    }
}

# Evaluation configuration
EVAL_CONFIG = {
    'top_k': [1, 3, 5],        # Top-k accuracy thresholds
    'num_visualizations': 10,   # Number of predictions to visualize
    'confidence_threshold': 0.5,  # Minimum confidence for visualization
    'plot_figsize': (12, 8),   # Figure size for plots
    'plot_dpi': 100,           # DPI for plots
}

# Visualization configuration
VIZ_CONFIG = {
    'plot_style': {
        'figsize': (12, 8),    # Default figure size
        'dpi': 100             # Default DPI
    },
    'font_size': {
        'title': 14,           # Title font size
        'label': 12,           # Axis label font size
        'tick': 10            # Tick label font size
    },
    'color_palette': 'husl',   # Color palette for plots
    'max_classes_plot': 20     # Maximum number of classes to show in plots
}