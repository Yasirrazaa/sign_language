"""Training script for sign language detection models."""

import torch
import logging
import wandb
from pathlib import Path
from typing import Dict, Optional, Union, Type

from ..models import (
    SignLanguageCNNLSTM,
    VideoTransformer,
    CNNLSTMConfig,
    TransformerConfig
)
from ..data import VideoDataset, create_dataloaders, get_class_weights
from .trainer import Trainer, TrainerConfig
from ..utils import get_checkpoint_dir

def train_model(
    model_type: str,
    video_data: Dict,
    class_mapping: Dict[str, int],
    config: Optional[Union[CNNLSTMConfig, TransformerConfig]] = None,
    trainer_config: Optional[TrainerConfig] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Train sign language detection model.
    
    Args:
        model_type: One of ['cnn_lstm', 'transformer']
        video_data: Video data dictionary
        class_mapping: Class name to index mapping
        config: Model configuration
        trainer_config: Training configuration
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on
        
    Returns:
        Training history
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    num_classes = len(class_mapping)
    if model_type == 'cnn_lstm':
        if config is None:
            config = CNNLSTMConfig(num_classes=num_classes)
        model = SignLanguageCNNLSTM(config)
    elif model_type == 'transformer':
        if config is None:
            config = TransformerConfig(num_classes=num_classes)
        model = VideoTransformer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        video_data=video_data,
        class_mapping=class_mapping
    )
    
    # Calculate class weights
    class_weights = get_class_weights(video_data, class_mapping)
    
    # Initialize trainer
    if trainer_config is None:
        trainer_config = TrainerConfig()
    
    trainer = Trainer(
        model=model,
        config=trainer_config
    )
    
    # Train model
    logger.info(f"Starting {model_type.upper()} training...")
    try:
        history = trainer.train(train_loader, val_loader)
        
        # Save final model
        final_path = get_checkpoint_dir() / f"{model_type}_final.pth"
        trainer.save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")
        
        return history
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return trainer.history
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def main():
    """Run training script."""
    import json
    from ..data import load_video_data
    
    # Load data
    video_data = load_video_data()
    
    # Load class mapping
    with open(Path(__file__).parent / 'class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    # Train CNN-LSTM model
    train_model(
        model_type='cnn_lstm',
        video_data=video_data,
        class_mapping=class_mapping
    )
    
    # Train Transformer model
    train_model(
        model_type='transformer',
        video_data=video_data,
        class_mapping=class_mapping
    )

if __name__ == '__main__':
    main()
