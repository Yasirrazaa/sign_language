"""Example script for training the memory-efficient sign language model."""

import torch
import logging
from pathlib import Path
import argparse
import yaml
from typing import Dict

from src.models.efficient_sign_net import create_efficient_sign_net
from src.training.efficient_trainer import create_trainer
from src.config.model_config import (
    EfficientSignNetConfig,
    MemoryOptimizedTrainingConfig,
    DataConfig,
    get_default_configs,
    validate_memory_settings
)
from src.data.loader import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_training(args):
    """Set up training components with memory optimization."""
    # Load configurations
    configs = get_default_configs()
    if args.config:
        custom_config = load_config(args.config)
        # Update default configs with custom ones
        for key, value in custom_config.items():
            if key in configs:
                configs[key].__dict__.update(value)
    
    # Validate memory settings
    warnings = validate_memory_settings(configs)
    for warning in warnings:
        logger.warning(warning)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders with memory optimization
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=configs['training'].batch_size,
        num_workers=configs['training'].num_workers,
        frame_size=configs['data'].frame_size
    )
    
    # Create model
    model = create_efficient_sign_net(
        num_classes=args.num_classes,
        in_channels=configs['model'].in_channels,
        base_channels=configs['model'].base_channels,
        num_frames=configs['model'].num_frames
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        config=configs['training'],
        device=device,
        checkpoint_dir=Path(args.output_dir) / 'checkpoints'
    )
    
    return trainer, configs

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train memory-efficient sign language model")
    
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--config', type=str,
                      help='Path to configuration file')
    parser.add_argument('--num-classes', type=int, default=100,
                      help='Number of sign classes')
    parser.add_argument('--cpu', action='store_true',
                      help='Use CPU instead of GPU')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up training
        trainer, configs = setup_training(args)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch, metrics = trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from epoch {start_epoch}")
            logger.info(f"Previous metrics: {metrics}")
        
        # Print memory optimization tips
        from src.config.model_config import print_memory_optimization_tips
        print_memory_optimization_tips()
        
        # Train model
        history = trainer.train(num_epochs=configs['training'].num_epochs)
        
        # Save training history
        history_file = output_dir / 'training_history.yml'
        with open(history_file, 'w') as f:
            yaml.dump(history, f)
        
        logger.info("Training completed successfully")
        logger.info(f"Training history saved to {history_file}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()