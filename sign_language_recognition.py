"""
Sign Language Recognition Pipeline

This script implements a complete pipeline for sign language recognition including:
1. Data Loading and Preprocessing
2. Data Analysis 
3. Model Training
4. Cross-Validation
5. Evaluation

Available architectures:
- CNN-LSTM (src/models/cnn_lstm.py)
- Video Transformer (src/models/video_transformer.py)
- I3D Model (WLASL/code/I3D/pytorch_i3d.py)
- TGCN Model (WLASL/code/TGCN/tgcn_model.py)
- EfficientSignNet (wlasl_modified/src/models/efficient_sign_net.py)
- Hybrid Models (wlasl_modified/src/models/hybrid_transformers.py):
  * CNN-Transformer
  * TimeSformer

Memory Management Tips:
1. Train one model at a time
2. Use appropriate batch sizes (start with 8)
3. Enable gradient checkpointing for transformer models
4. Clean up memory between training runs
5. Monitor GPU memory usage
"""

# Standard imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import logging
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model architectures
try:
    from src.models.cnn_lstm import SignLanguageCNNLSTM, CNNLSTMConfig
    from src.models.video_transformer import VideoTransformer, TransformerConfig
    from WLASL.code.I3D.pytorch_i3d import InceptionI3d
    from wlasl_modified.src.models.efficient_sign_net import EfficientSignNet
    from wlasl_modified.src.models.hybrid_transformers import CNNTransformer, TimeSformer
except ImportError as e:
    logger.warning(f"Some models could not be imported: {e}")

from wlasl_modified.src.data.preprocessing import MemoryEfficientPreprocessor
from wlasl_modified.src.data.loader import create_data_loaders, MemoryEfficientDataset
from sklearn.model_selection import KFold

class DataProcessor:
    """Handle data loading and preprocessing."""
    
    def __init__(self, data_dir: Path, processed_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.preprocessor = MemoryEfficientPreprocessor(
            output_dir=processed_dir / 'frames',
            frame_size=(224, 224),  # Standard size for most models
            target_fps=25,  # Standard frame rate
            chunk_size=32   # Process videos in chunks to save memory
        )
    
    def preprocess_videos(self):
        """Preprocess all videos in data directory."""
        # Create output directory if needed
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_paths = list(self.data_dir.glob('**/*.mp4'))
        logger.info(f"Found {len(video_paths)} videos")
        
        # Process each video
        results = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                result = self.preprocessor.preprocess_video(video_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
        
        return results

    def create_dataloaders(self, batch_size: int = 8, num_workers: int = 4):
        """Create train/val/test dataloaders."""
        # Load data info
        with open(self.processed_dir / 'preprocessing_results.json', 'r') as f:
            data_info = json.load(f)

        return create_data_loaders(
            data_info=data_info,
            processed_dir=self.processed_dir / 'frames',
            batch_size=batch_size,
            num_workers=num_workers
        )

class DataAnalyzer:
    """Analyze processed dataset."""
    
    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
    
    def analyze_dataset(self):
        """Analyze dataset statistics."""
        frame_dirs = list((self.processed_dir / 'frames').glob('*'))
        
        stats = {
            'num_samples': len(frame_dirs),
            'frames_per_video': [],
            'video_sizes': [],
            'class_distribution': {}
        }
        
        for frame_dir in frame_dirs:
            # Count frames
            frames = list(frame_dir.glob('*.jpg'))
            stats['frames_per_video'].append(len(frames))
            
            # Calculate video size
            size = sum(f.stat().st_size for f in frames) / 1024**2  # MB
            stats['video_sizes'].append(size)
            
            # Update class distribution
            class_name = frame_dir.name.split('_')[0]
            stats['class_distribution'][class_name] = \
                stats['class_distribution'].get(class_name, 0) + 1
        
        return stats
    
    def plot_statistics(self, stats: dict):
        """Plot dataset statistics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Frames per video distribution
        sns.histplot(stats['frames_per_video'], ax=ax1)
        ax1.set_title('Frames per Video Distribution')
        ax1.set_xlabel('Number of Frames')
        
        # Video sizes distribution
        sns.histplot(stats['video_sizes'], ax=ax2)
        ax2.set_title('Video Sizes Distribution')
        ax2.set_xlabel('Size (MB)')
        
        # Class distribution
        class_dist = pd.Series(stats['class_distribution']).sort_values(ascending=False)
        class_dist.plot(kind='bar', ax=ax3)
        ax3.set_title('Class Distribution')
        ax3.set_xlabel('Class')
        plt.xticks(rotation=45)
        
        # Basic statistics
        ax4.axis('off')
        stats_text = (
            f"Total samples: {stats['num_samples']}\n"
            f"Number of classes: {len(stats['class_distribution'])}\n"
            f"Avg frames per video: {np.mean(stats['frames_per_video']):.1f}\n"
            f"Avg video size: {np.mean(stats['video_sizes']):.1f} MB"
        )
        ax4.text(0.1, 0.5, stats_text, fontsize=12)
        
        plt.tight_layout()
        plt.show()

class ModelTrainer:
    """Handle model training and evaluation."""
    
    AVAILABLE_MODELS = {
        'cnn_lstm': 'CNN-LSTM model with ResNet backbone',
        'transformer': 'Video Transformer with self-attention',
        'i3d': 'Inflated 3D ConvNet (I3D)',
        'tgcn': 'Temporal Graph Convolutional Network',
        'efficient': 'Memory-efficient Sign Language Network',
        'cnn_transformer': 'Hybrid CNN-Transformer',
        'timesformer': 'Time-Space Transformer'
    }
    
    def __init__(self, 
                 model_name: str,
                 num_classes: int,
                 device: torch.device = None):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )
    
    def _create_model(self):
        """Create model based on specified architecture."""
        if self.model_name == 'cnn_lstm':
            config = CNNLSTMConfig(num_classes=self.num_classes)
            return SignLanguageCNNLSTM(config)
        
        elif self.model_name == 'transformer':
            config = TransformerConfig(num_classes=self.num_classes)
            return VideoTransformer(config)
        
        elif self.model_name == 'i3d':
            return InceptionI3d(
                num_classes=self.num_classes,
                in_channels=3
            )
        
        elif self.model_name == 'tgcn':
            return TGCN(
                num_classes=self.num_classes,
                in_channels=3,
                graph_args={'layout': 'openpose', 'strategy': 'spatial'}
            )
        
        elif self.model_name == 'efficient':
            return EfficientSignNet(
                num_classes=self.num_classes,
                in_channels=3
            )
        
        elif self.model_name == 'cnn_transformer':
            return CNNTransformer(
                num_classes=self.num_classes,
                num_frames=30
            )
        
        elif self.model_name == 'timesformer':
            return TimeSformer(
                num_classes=self.num_classes,
                num_frames=30
            )
        
        else:
            raise ValueError(f"Model creation not implemented for: {self.model_name}")
    
    @classmethod
    def list_available_models(cls):
        """Display all available model architectures."""
        print("Available Model Architectures:")
        for name, desc in cls.AVAILABLE_MODELS.items():
            print(f"- {name}: {desc}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f'Train Batch: {batch_idx} Loss: {loss.item():.4f}')
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        # Update scheduler
        self.scheduler.step(val_loss)
        
        return {
            'val_loss': val_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Full training loop with validation."""
        best_val_acc = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_accuracy'])
            
            # Save best model
            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                torch.save(self.model.state_dict(), 
                          f'checkpoints/{self.model_name}_best.pth')
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['accuracy']:.4f} "
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )
        
        return history

class CrossValidator:
    """Handle k-fold cross-validation."""
    
    def __init__(self, 
                 model_name: str,
                 num_classes: int,
                 n_splits: int = 5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def run_cross_validation(self, dataset, batch_size=8):
        """Run k-fold cross-validation."""
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(dataset)):
            logger.info(f"\nFold {fold + 1}/{self.n_splits}")
            
            # Create data loaders for this fold
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=val_sampler
            )
            
            # Create and train model
            trainer = ModelTrainer(self.model_name, self.num_classes)
            history = trainer.train(train_loader, val_loader)
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': max(history['val_acc']),
                'final_train_loss': history['train_loss'][-1],
                'history': history
            })
            
            # Clean up to free memory
            del trainer
            torch.cuda.empty_cache()
        
        return fold_results

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_cv_results(cv_results):
    """Plot cross-validation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracies across folds
    accuracies = [result['best_val_acc'] for result in cv_results]
    ax1.bar(range(1, len(accuracies) + 1), accuracies)
    ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.4f}')
    ax1.set_title('Cross-Validation Accuracies')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Best Validation Accuracy')
    ax1.legend()
    
    # Plot training curves
    for result in cv_results:
        ax2.plot(result['history']['val_acc'],
                label=f"Fold {result['fold']}")
    ax2.set_title('Validation Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = Path('video')
    PROCESSED_DIR = Path('processed')
    NUM_CLASSES = 26  # Number of sign classes
    BATCH_SIZE = 8
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    
    # List available models
    ModelTrainer.list_available_models()
    
    # 1. Data Processing
    processor = DataProcessor(DATA_DIR, PROCESSED_DIR)
    
    # Process videos if not already done
    if not (PROCESSED_DIR / 'frames').exists():
        processor.preprocess_videos()
    
    # Create dataloaders
    data_loaders = processor.create_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Data Analysis
    analyzer = DataAnalyzer(PROCESSED_DIR)
    stats = analyzer.analyze_dataset()
    analyzer.plot_statistics(stats)
    
    # 3. Model Training and Evaluation
    MODEL_NAME = 'cnn_lstm'  # Change this to try different architectures
    
    # Create trainer
    trainer = ModelTrainer(MODEL_NAME, NUM_CLASSES)
    
    # Train model
    history = trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        num_epochs=50
    )
    
    # Plot training history
    plot_training_history(history)
    
    # 4. Cross-Validation
    validator = CrossValidator(MODEL_NAME, NUM_CLASSES)
    cv_results = validator.run_cross_validation(data_loaders['train'].dataset)
    
    # Print and plot cross-validation results
    accuracies = [result['best_val_acc'] for result in cv_results]
    print(f"\nCross-validation results for {MODEL_NAME}:")
    print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    plot_cv_results(cv_results)

if __name__ == "__main__":
    main()