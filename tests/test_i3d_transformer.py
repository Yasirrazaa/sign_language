"""Tests for I3D-Transformer models."""

import torch
import pytest
from pathlib import Path
import numpy as np
from torch.utils.data import TensorDataset
from src.training.trainer import MemoryEfficientTrainer, TrainerConfig
from src.models.i3d_transformer import I3DTransformer, I3DTimeSformer, create_model

@pytest.fixture
def dummy_input():
    """Create dummy input tensor."""
    return torch.randn(2, 3, 32, 224, 224)

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_model_creation(model_class):
    """Test model initialization."""
    model = model_class(
        num_classes=100,
        num_frames=32,
        embed_dim=768,
        depth=4,
        num_heads=8
    )
    assert isinstance(model, model_class)

@pytest.mark.parametrize("model_name", [
    'i3d_transformer',
    'i3d_timesformer'
])
def test_model_factory(model_name):
    """Test model creation through factory function."""
    model = create_model(
        model_name=model_name,
        num_classes=100,
        num_frames=32
    )
    assert isinstance(model, (I3DTransformer, I3DTimeSformer))

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_forward_pass(model_class, dummy_input):
    """Test forward pass with dummy input."""
    model = model_class(
        num_classes=100,
        num_frames=32,
        embed_dim=768,
        depth=4,
        num_heads=8
    )
    
    output = model(dummy_input)
    assert output.shape == (2, 100)

def test_invalid_model_name():
    """Test error handling for invalid model name."""
    with pytest.raises(ValueError):
        create_model('invalid_model', num_classes=100, num_frames=32)

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_gradient_checkpointing(model_class, dummy_input):
    """Test gradient checkpointing functionality."""
    model = model_class(
        num_classes=100,
        num_frames=32,
        use_checkpoint=True
    )
    
    # Test with checkpointing enabled
    model.train()
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # Test with checkpointing disabled
    model.use_checkpoint = False
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_class,expected_ratio", [
    (I3DTransformer, 0.8),
    (I3DTimeSformer, 0.8)
])
def test_memory_efficiency(model_class, expected_ratio, dummy_input):
    """Test memory efficiency of models."""
    model = model_class(
        num_classes=100,
        num_frames=32,
        use_checkpoint=True
    ).cuda()
    dummy_input = dummy_input.cuda()
    
    # Memory usage without checkpointing
    model.use_checkpoint = False
    torch.cuda.reset_peak_memory_stats()
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    memory_without_checkpoint = torch.cuda.max_memory_allocated()
    
    # Memory usage with checkpointing
    model.use_checkpoint = True
    torch.cuda.reset_peak_memory_stats()
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    memory_with_checkpoint = torch.cuda.max_memory_allocated()
    
    # Memory ratio should be less than expected_ratio
    memory_ratio = memory_with_checkpoint / memory_without_checkpoint
    assert memory_ratio < expected_ratio, f"Memory efficiency not meeting target: {memory_ratio:.2f}"

def test_i3d_pretrained_loading(tmp_path):
    """Test loading pretrained I3D weights."""
    # Create dummy weights
    dummy_state = {'conv3d_0a_1x1.conv3d.weight': torch.randn(64, 3, 1, 1, 1)}
    dummy_weights_path = tmp_path / "dummy_i3d.pt"
    torch.save(dummy_state, dummy_weights_path)
    
    # Test loading
    model = I3DTransformer(
        num_classes=100,
        num_frames=32,
        i3d_pretrained_path=str(dummy_weights_path)
    )
    assert isinstance(model, I3DTransformer)

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_training_compatibility(model_class, tmp_path):
    """Test compatibility with training infrastructure."""
    model = model_class(
        num_classes=100,
        num_frames=32,
        use_checkpoint=True
    )
    
    config = TrainerConfig(
        num_epochs=1,
        checkpoint_dir=tmp_path,
        mixed_precision=True,
        enable_checkpointing=True
    )
    
    trainer = MemoryEfficientTrainer(model, config)
    assert trainer.model == model
    assert trainer.scaler is not None
    assert model.use_checkpoint

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_output_range(model_class, dummy_input):
    """Test model output range and validity."""
    model = model_class(
        num_classes=100,
        num_frames=32
    ).eval()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output range
    assert not torch.isnan(output).any(), "Model produced NaN values"
    assert not torch.isinf(output).any(), "Model produced infinite values"

@pytest.mark.parametrize("model_class", [
    I3DTransformer,
    I3DTimeSformer
])
def test_batch_independence(model_class, dummy_input):
    """Test batch independence of model outputs."""
    model = model_class(
        num_classes=100,
        num_frames=32
    ).eval()
    
    # Process first sample
    with torch.no_grad():
        output1 = model(dummy_input[0:1])
        output2 = model(dummy_input[1:2])
        output_batch = model(dummy_input)[0:2]
    
    # Check if individual processing matches batch processing
    assert torch.allclose(torch.cat([output1, output2]), output_batch, rtol=1e-5)