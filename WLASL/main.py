"""Main script demonstrating usage of the modified WLASL framework."""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import logging
from datetime import datetime

from src.preprocessing.video_processor import VideoProcessor, BatchVideoProcessor
from src.data.data_loader import create_data_loaders
from src.training.trainer import Trainer
from src.training.cross_validate import CrossValidator
from configs.base_config import (
    DATA_CONFIG,
    I3D_CONFIG,
    TGCN_CONFIG,
    TRAIN_CONFIG
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_arg_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description='WLASL Training Pipeline')
    parser.add_argument('--model', type=str, choices=['i3d', 'tgcn'], default='i3d',
                      help='Model architecture to use')
    parser.add_argument('--preprocess', action='store_true',
                      help='Run preprocessing on videos')
    parser.add_argument('--cross-validate', action='store_true',
                      help='Perform cross-validation')
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=DATA_CONFIG['batch_size'],
                      help='Batch size for training')
    return parser

def preprocess_videos(data_path: Path):
    """Preprocess videos with memory-efficient processing."""
    logger.info("Starting video preprocessing...")
    
    video_processor = VideoProcessor(
        frame_size=DATA_CONFIG['frame_size'],
        num_frames=DATA_CONFIG['num_frames'],
        fps=DATA_CONFIG['fps']
    )
    
    batch_processor = BatchVideoProcessor(video_processor)
    video_paths = list(Path(data_path).glob('**/*.mp4'))
    
    if not video_paths:
        raise ValueError(f"No video files found in {data_path}")
        
    logger.info(f"Found {len(video_paths)} videos to process")
    batch_processor.process_batch(video_paths)
    logger.info("Video preprocessing completed")

def train_model(model_name: str, data_info: list, cross_validate: bool = False):
    """Train the selected model with optional cross-validation."""
    # Setup model configuration
    if model_name == 'i3d':
        from code.I3D.pytorch_i3d import InceptionI3d
        model_class = InceptionI3d
        model_params = I3D_CONFIG
    else:  # tgcn
        from code.TGCN.tgcn_model import TGCN
        model_class = TGCN
        model_params = TGCN_CONFIG
    
    # Update number of classes
    num_classes = len(set(item['label'] for item in data_info))
    model_params['num_classes'] = num_classes
    
    if cross_validate:
        logger.info(f"Starting {TRAIN_CONFIG['num_folds']}-fold cross-validation")
        validator = CrossValidator(
            model_class=model_class,
            model_params=model_params,
            data_info=data_info,
            criterion=nn.CrossEntropyLoss(),
            num_folds=TRAIN_CONFIG['num_folds']
        )
        results = validator.run()
        logger.info("Cross-validation completed")
        logger.info("Aggregate Results:")
        for metric, values in results['aggregate_metrics'].items():
            logger.info(f"{metric}: {values['mean']:.2f} ± {values['std']:.2f}")
    else:
        logger.info("Starting single model training")
        # Create data loaders without cross-validation
        dataloaders = create_data_loaders(data_info)
        
        # Initialize model
        model = model_class(**model_params)
        
        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=TRAIN_CONFIG['reduce_lr_factor'],
            patience=TRAIN_CONFIG['reduce_lr_patience'],
            min_lr=TRAIN_CONFIG['min_learning_rate']
        )
        
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        history = trainer.train()
        logger.info("Training completed")

def main():
    """Main function to run the pipeline."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Update batch size if provided
    if args.batch_size != DATA_CONFIG['batch_size']:
        DATA_CONFIG['batch_size'] = args.batch_size
    
    # Create data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Preprocess videos if requested
    if args.preprocess:
        preprocess_videos(data_path)
    
    # Load data info
    data_info_path = data_path / 'data_info.json'
    if not data_info_path.exists():
        raise ValueError(f"Data info file not found: {data_info_path}")
        
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    
    # Train model
    train_model(args.model, data_info, args.cross_validate)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)