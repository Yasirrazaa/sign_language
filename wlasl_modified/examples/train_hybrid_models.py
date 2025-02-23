"""Example script for training hybrid transformer models."""

import torch
import logging
from pathlib import Path
import argparse
import yaml
import os
from datetime import datetime

from src.models.hybrid_transformers import create_model
from src.training.hybrid_trainers import create_trainer
from configs.hybrid_transformer_config import get_config, print_memory_recommendations
from src.data.loader import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_wandb(config: dict, model_name: str):
    """Setup Weights & Biases logging."""
    try:
        import wandb
        wandb.init(
            project="wlasl-hybrid-transformers",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping experiment tracking.")
        return None

def train_model(args):
    """Train hybrid transformer model."""
    # Get configurations
    config = get_config(args.model)
    
    # Override config with command line arguments
    if args.batch_size:
        config['trainer'].batch_size = args.batch_size
    if args.epochs:
        config['trainer'].num_epochs = args.epochs
    if args.learning_rate:
        config['trainer'].learning_rate = args.learning_rate
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Print memory recommendations
    print_memory_recommendations()
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config['trainer'].batch_size,
        num_workers=config['trainer'].num_workers,
        **vars(config['data'])
    )
    
    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        num_frames=config['data'].num_frames,
        **vars(config['model'])
    )
    
    # Create optimizer and scheduler
    optimizer = config['trainer'].get_optimizer(model)
    scheduler = config['trainer'].get_scheduler(optimizer)
    
    # Create trainer
    trainer = create_trainer(
        model_name=args.model,
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=device,
        config=vars(config['trainer']),
        checkpoint_dir=Path(args.output_dir) / 'checkpoints' / args.model,
        scheduler=scheduler
    )
    
    # Setup wandb logging
    wandb = setup_wandb(config, args.model)
    
    try:
        logger.info("Starting training...")
        best_val_acc = 0.0
        
        for epoch in range(config['trainer'].num_epochs):
            # Training phase
            train_metrics = trainer.train_epoch()
            
            # Validation phase
            val_metrics = trainer.validate()
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            logger.info(
                f"Epoch {epoch + 1}/{config['trainer'].num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Accuracy: {val_metrics['val_accuracy']:.4f}"
            )
            
            if wandb is not None:
                wandb.log(metrics)
            
            # Save checkpoint
            is_best = val_metrics['val_accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['val_accuracy']
            trainer.save_checkpoint(metrics, is_best)
            
            # Early stopping
            if trainer.patience_counter >= config['trainer'].patience:
                logger.info("Early stopping triggered")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    finally:
        if wandb is not None:
            wandb.finish()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train hybrid transformer models")
    
    parser.add_argument('--model', type=str, required=True,
                      choices=['cnn_transformer', 'timesformer'],
                      help='Model architecture to use')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--num-classes', type=int, required=True,
                      help='Number of classes')
    
    # Optional training arguments
    parser.add_argument('--batch-size', type=int,
                      help='Override batch size from config')
    parser.add_argument('--epochs', type=int,
                      help='Override number of epochs from config')
    parser.add_argument('--learning-rate', type=float,
                      help='Override learning rate from config')
    parser.add_argument('--cpu', action='store_true',
                      help='Use CPU instead of GPU')
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_model(args)

if __name__ == '__main__':
    main()