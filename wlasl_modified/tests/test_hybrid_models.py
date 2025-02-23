"""Unit tests for hybrid transformer models."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import gc

from src.models.hybrid_transformers import (
    CNNTransformer,
    TimeSformer,
    create_model,
    MemoryEfficientAttention
)
from src.training.hybrid_trainers import (
    CNNTransformerTrainer,
    TimeSformerTrainer,
    create_trainer
)
from configs.hybrid_transformer_config import get_config

@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    batch_size = 2
    num_frames = 30
    height = 224
    width = 224
    channels = 3
    
    frames = torch.randn(batch_size, num_frames, channels, height, width)
    labels = torch.randint(0, 26, (batch_size,))
    
    return frames, labels

@pytest.fixture
def test_config():
    """Get test configuration."""
    config = get_config('cnn_transformer')
    config['trainer'].batch_size = 2
    config['trainer'].num_epochs = 2
    return config

def test_memory_efficient_attention():
    """Test memory-efficient attention implementation."""
    batch_size = 2
    seq_len = 16
    dim = 64
    num_heads = 4
    
    # Create attention module
    attn = MemoryEfficientAttention(dim=dim, num_heads=num_heads)
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Record initial memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    output = attn(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Check memory usage
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated()
        memory_diff = current_memory - initial_memory
        
        # Memory usage should be bounded
        assert memory_diff < dim * seq_len * batch_size * 4 * 2  # Rough estimate

def test_cnn_transformer():
    """Test CNN-Transformer model."""
    model = CNNTransformer(
        num_classes=26,
        num_frames=30,
        embed_dim=64,
        depth=2,
        num_heads=2
    )
    
    # Test forward pass
    batch = torch.randn(2, 30, 3, 224, 224)
    output = model(batch)
    
    assert output.shape == (2, 26)
    assert not torch.isnan(output).any()
    
    # Test gradient checkpointing
    model.use_checkpoint = True
    output = model(batch)
    assert output.shape == (2, 26)

def test_timesformer():
    """Test TimeSformer model."""
    model = TimeSformer(
        num_classes=26,
        num_frames=30,
        img_size=224,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2
    )
    
    # Test forward pass
    batch = torch.randn(2, 30, 3, 224, 224)
    output = model(batch)
    
    assert output.shape == (2, 26)
    assert not torch.isnan(output).any()
    
    # Test divided space-time attention
    model = TimeSformer(
        num_classes=26,
        num_frames=30,
        img_size=224,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        divided_space_time=True
    )
    output = model(batch)
    assert output.shape == (2, 26)

def test_model_creation():
    """Test model creation utility."""
    # Test CNN-Transformer creation
    model = create_model(
        model_name='cnn_transformer',
        num_classes=26,
        num_frames=30
    )
    assert isinstance(model, CNNTransformer)
    
    # Test TimeSformer creation
    model = create_model(
        model_name='timesformer',
        num_classes=26,
        num_frames=30
    )
    assert isinstance(model, TimeSformer)
    
    # Test invalid model name
    with pytest.raises(ValueError):
        create_model('invalid_model', num_classes=26, num_frames=30)

def test_trainer_creation(sample_batch, test_config):
    """Test trainer creation and basic training loop."""
    frames, labels = sample_batch
    
    # Create simple dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return frames[0], labels[0]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    val_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    
    # Test CNN-Transformer trainer
    model = create_model('cnn_transformer', num_classes=26, num_frames=30)
    trainer = create_trainer(
        model_name='cnn_transformer',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device('cpu'),
        config=test_config['trainer'],
        checkpoint_dir=Path('tests/checkpoints')
    )
    assert isinstance(trainer, CNNTransformerTrainer)
    
    # Test training step
    metrics = trainer.train_epoch()
    assert 'loss' in metrics
    assert 'accuracy' in metrics

def test_memory_cleanup(sample_batch, test_config):
    """Test memory cleanup during training."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    frames, labels = sample_batch
    frames = frames.cuda()
    labels = labels.cuda()
    
    # Create model and trainer
    model = create_model('cnn_transformer', num_classes=26, num_frames=30).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Forward and backward pass
    outputs = model(frames)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Check memory after cleanup
    gc.collect()
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    
    # Memory should be cleaned up
    assert final_memory <= initial_memory * 1.1  # Allow small overhead

def test_gradient_accumulation(sample_batch, test_config):
    """Test gradient accumulation in trainers."""
    frames, labels = sample_batch
    
    # Create model and trainer
    model = create_model('cnn_transformer', num_classes=26, num_frames=30)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Single large batch
    outputs = model(frames)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Get parameters after single batch
    single_batch_params = [p.clone() for p in model.parameters()]
    
    # Reset model
    model = create_model('cnn_transformer', num_classes=26, num_frames=30)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Accumulated gradients
    for i in range(2):
        half_frames = frames[i:i+1]
        half_labels = labels[i:i+1]
        outputs = model(half_frames)
        loss = nn.CrossEntropyLoss()(outputs, half_labels) / 2  # Scale loss
        loss.backward()
    
    optimizer.step()
    accumulated_params = [p.clone() for p in model.parameters()]
    
    # Parameters should be similar
    for s, a in zip(single_batch_params, accumulated_params):
        assert torch.allclose(s, a, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])