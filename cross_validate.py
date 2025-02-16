"""Cross validation utilities for Sign Language Detection using PyTorch."""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Any, Tuple
import logging
import json
from torch.utils.data import DataLoader, Subset
import wandb
from collections import defaultdict

from src.config import TRAIN_CONFIG, CHECKPOINT_DIR, MODEL_CONFIG
from data_loader import VideoDataset
from models.video_transformer import VideoTransformer, TransformerConfig
from models.cnn_lstm import SignLanguageCNNLSTM, CNNLSTMConfig
from train import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_fold_datasets(
    dataset: VideoDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for a fold.
    
    Args:
        dataset: Full dataset
        train_idx: Training indices
        val_idx: Validation indices
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def evaluate_fold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = defaultdict(float)
    total_samples = 0
    
    with torch.no_grad():
        for frames, (labels, bboxes) in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            
            # Get predictions
            class_pred, bbox_pred = model(frames)
            
            # Calculate metrics
            pred = torch.argmax(class_pred, dim=1)
            true = torch.argmax(labels, dim=1)
            
            # Accuracy
            correct = (pred == true).float()
            metrics['accuracy'] += correct.sum().item()
            
            # Top-k accuracy
            _, top_k = torch.topk(class_pred, k=5, dim=1)
            for k in [1, 3, 5]:
                top_k_correct = torch.any(top_k[:, :k] == true.unsqueeze(1), dim=1).float()
                metrics[f'top_{k}_accuracy'] += top_k_correct.sum().item()
            
            # Bounding box error
            bbox_error = torch.nn.functional.mse_loss(bbox_pred, bboxes, reduction='sum')
            metrics['bbox_mse'] += bbox_error.item()
            
            total_samples += frames.size(0)
    
    # Average metrics
    for metric in metrics:
        metrics[metric] /= total_samples
    
    return dict(metrics)

def cross_validate(
    video_data: List[Dict],
    class_mapping: Dict[str, int],
    model_type: str = 'transformer',
    device: torch.device = None
) -> List[Dict]:
    """
    Perform stratified k-fold cross validation.
    
    Args:
        video_data: List of preprocessed video data
        class_mapping: Class name to index mapping
        model_type: Type of model to train ('cnn_lstm' or 'transformer')
        device: Device to use for training
        
    Returns:
        List of metrics for each fold
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(TRAIN_CONFIG['random_seed'])
    torch.cuda.manual_seed_all(TRAIN_CONFIG['random_seed'])
    np.random.seed(TRAIN_CONFIG['random_seed'])
    
    # Create dataset
    dataset = VideoDataset(video_data, class_mapping)
    
    # Get labels for stratification
    labels = []
    for data in video_data:
        if data['split'] == 'train':
            labels.append(class_mapping[data['gloss']])
    labels = np.array(labels)
    
    # Create folds
    skf = StratifiedKFold(
        n_splits=TRAIN_CONFIG['num_folds'],
        shuffle=True,
        random_state=TRAIN_CONFIG['random_seed']
    )
    
    # Track metrics
    all_metrics = []
    
    # Get indices of training data
    train_indices = np.array([
        i for i, v in enumerate(video_data)
        if v['split'] == 'train'
    ])
    
    # Create model directory
    model_dir = CHECKPOINT_DIR / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize wandb
        wandb.init(
            project="sign-language-detection",
            config={
                "model_type": model_type,
                "train_config": TRAIN_CONFIG,
                "model_config": MODEL_CONFIG[model_type.lower()],
            }
        )
        
        # Train each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_indices, labels), 1):
            logger.info(f"\nTraining fold {fold}/{TRAIN_CONFIG['num_folds']}")
            
            # Create data loaders
            train_loader, val_loader = create_fold_datasets(
                dataset,
                train_indices[train_idx],
                train_indices[val_idx],
                TRAIN_CONFIG['batch_size']
            )
            
            # Create model and config
            num_classes = len(class_mapping)
            if model_type == 'transformer':
                model = VideoTransformer(num_classes)
                config = TransformerConfig(num_classes)
            else:
                model = SignLanguageCNNLSTM(num_classes)
                config = CNNLSTMConfig(num_classes)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                fold=fold,
                model_type=model_type
            )
            
            # Train fold
            trainer.train()
            
            # Load best model
            best_model_path = trainer.checkpoint_dir / 'best_model.pth'
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate fold
            metrics = evaluate_fold(model, val_loader, device)
            metrics['fold'] = fold
            metrics['best_epoch'] = checkpoint['epoch']
            all_metrics.append(metrics)
            
            # Save fold metrics
            fold_dir = model_dir / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)
            with open(fold_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Log results
            logger.info(f"Fold {fold} Results:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                    wandb.log({f"fold_{fold}/{metric}": value})
    
    except KeyboardInterrupt:
        logger.info("Cross-validation interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise
    
    finally:
        # Calculate and save average metrics
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            if isinstance(all_metrics[0][metric], (int, float)):
                values = [m[metric] for m in all_metrics]
                avg = float(np.mean(values))
                std = float(np.std(values))
                avg_metrics[f'avg_{metric}'] = avg
                avg_metrics[f'std_{metric}'] = std
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        f"final/{metric}_mean": avg,
                        f"final/{metric}_std": std
                    })
        
        with open(model_dir / 'average_metrics.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        # Close wandb
        if wandb.run is not None:
            wandb.finish()
    
    return all_metrics

if __name__ == '__main__':
    # Load data
    with open('processed/preprocessing_results.json', 'r') as f:
        video_data = json.load(f)
    
    with open('processed/analysis/selected_classes.json', 'r') as f:
        selected_classes = json.load(f)
        class_mapping = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run cross-validation for both models
    for model_type in ['transformer', 'cnn_lstm']:
        logger.info(f"\nRunning cross-validation for {model_type} model")
        metrics = cross_validate(video_data, class_mapping, model_type, device)
        
        # Print final results
        avg_metrics = {}
        for metric in metrics[0].keys():
            if isinstance(metrics[0][metric], (int, float)):
                values = [m[metric] for m in metrics]
                avg = np.mean(values)
                std = np.std(values)
                avg_metrics[metric] = f"{avg:.4f} ± {std:.4f}"
        
        logger.info("\nFinal Results:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value}")
