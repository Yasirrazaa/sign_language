"""Tests for memory-efficient cross-validation."""

import torch
import pytest
from pathlib import Path
import numpy as np
from torch.utils.data import TensorDataset

from src.training.cross_validate import MemoryEfficientCrossValidator
from src.training.trainer import TrainerConfig
from src.models import VideoTransformer, CNNTransformer, TimeSformer, I3DTransformer

def create_dummy_dataset(num_samples=100):
    """Create dummy dataset for testing."""
    frames = torch.randn(num_samples, 3, 32, 224, 224)
    labels = torch.randint(0, 10, (num_samples,))
    labels_one_hot = torch.zeros(num_samples, 10)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return TensorDataset(frames, labels_one_hot)

@pytest.mark.parametrize("model_class", [
    VideoTransformer,
    CNNTransformer,
    TimeSformer,
    I3DTransformer
])
def test_cross_validator_initialization(model_class, tmp_path):
    """Test cross-validator initialization with different models."""
    model_config = {
        'num_classes': 10,
        'num_frames': 32,
        'embed_dim': 512,
        'use_checkpoint': True
    }
    
    trainer_config = TrainerConfig(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        mixed_precision=True,
        enable_checkpointing=True
    )
    
    validator = MemoryEfficientCrossValidator(
        model_class=model_class,
        model_config=model_config,
        trainer_config=trainer_config,
        num_folds=3
    )
    
    assert validator.model_class == model_class
    assert validator.num_folds == 3
    assert isinstance(validator.fold_histories, list)
    assert isinstance(validator.fold_metrics, list)

@pytest.mark.parametrize("model_class", [
    VideoTransformer,
    CNNTransformer,
    TimeSformer,
    I3DTransformer
])
def test_single_fold_training(model_class, tmp_path):
    """Test training a single fold with each model type."""
    # Create configurations
    model_config = {
        'num_classes': 10,
        'num_frames': 32,
        'embed_dim': 512,
        'use_checkpoint': True
    }
    
    trainer_config = TrainerConfig(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        mixed_precision=True,
        enable_checkpointing=True
    )
    
    # Initialize cross-validator
    validator = MemoryEfficientCrossValidator(
        model_class=model_class,
        model_config=model_config,
        trainer_config=trainer_config,
        num_folds=3
    )
    
    # Create dummy dataset
    dataset = create_dummy_dataset(num_samples=50)
    
    # Create data loaders for a single fold
    train_size = 40
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, 50))
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # Train single fold
    results = validator.train_fold(0, train_loader, val_loader, tmp_path)
    
    # Check results
    assert 'history' in results
    assert 'metrics' in results
    assert isinstance(results['history'], dict)
    assert isinstance(results['metrics'], dict)
    assert 'val_loss' in results['metrics']
    assert 'val_accuracy' in results['metrics']

@pytest.mark.parametrize("model_class", [
    VideoTransformer,
    CNNTransformer,
    TimeSformer,
    I3DTransformer
])
def test_full_cross_validation(model_class, tmp_path):
    """Test full cross-validation process."""
    model_config = {
        'num_classes': 10,
        'num_frames': 32,
        'embed_dim': 512,
        'use_checkpoint': True
    }
    
    trainer_config = TrainerConfig(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        mixed_precision=True,
        enable_checkpointing=True
    )
    
    # Initialize cross-validator
    validator = MemoryEfficientCrossValidator(
        model_class=model_class,
        model_config=model_config,
        trainer_config=trainer_config,
        num_folds=2  # Use 2 folds for faster testing
    )
    
    # Create dummy dataset
    dataset = create_dummy_dataset(num_samples=40)
    
    # Run cross-validation
    results = validator.cross_validate(
        dataset=dataset,
        batch_size=4,
        log_dir=tmp_path
    )
    
    # Check results
    assert isinstance(results, dict)
    assert len(validator.fold_metrics) == 2
    assert len(validator.fold_histories) == 2
    
    # Check if metrics were properly aggregated
    assert 'accuracy' in results
    assert 'mean' in results['accuracy']
    assert 'std' in results['accuracy']
    assert 'ci_lower' in results['accuracy']
    assert 'ci_upper' in results['accuracy']
    assert isinstance(results['accuracy']['values'], list)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_class", [
    VideoTransformer,
    CNNTransformer,
    TimeSformer,
    I3DTransformer
])
def test_memory_efficiency(model_class, tmp_path):
    """Test memory efficiency of cross-validation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model_config = {
        'num_classes': 10,
        'num_frames': 32,
        'embed_dim': 512,
        'use_checkpoint': True
    }
    
    trainer_config = TrainerConfig(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        mixed_precision=True,
        enable_checkpointing=True
    )
    
    # Initialize cross-validator
    validator = MemoryEfficientCrossValidator(
        model_class=model_class,
        model_config=model_config,
        trainer_config=trainer_config,
        num_folds=2
    )
    
    # Create dummy dataset
    dataset = create_dummy_dataset(num_samples=40)
    
    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.max_memory_allocated()
    
    # Run cross-validation
    _ = validator.cross_validate(
        dataset=dataset,
        batch_size=4,
        log_dir=tmp_path
    )
    
    # Check peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    current_memory = torch.cuda.memory_allocated()
    
    # Memory should be cleared between folds
    assert current_memory < peak_memory, "Memory not properly cleared between folds"
    
    # Check if results were saved
    assert (tmp_path / "aggregate_results.json").exists()
    assert (tmp_path / "fold_1_results.json").exists()
    assert (tmp_path / "fold_2_results.json").exists()