"""Data loading and preprocessing utilities using PyTorch."""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import json
import cv2
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from ..config import DATA_CONFIG
from ..utils import get_processed_dir

def load_video_data() -> List[Dict]:
    """
    Load preprocessed video data.
    
    Returns:
        List of video data dictionaries
    """
    # Check both possible file names for backward compatibility
    possible_paths = [
        get_processed_dir() / 'dataset.json',
        get_processed_dir() / 'preprocessing_results.json'
    ]
    
    for results_path in possible_paths:
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(
        "Preprocessing results not found. Run preprocessing first."
    )

class VideoDataset(Dataset):
    """Memory-efficient dataset handler for video processing."""
    
    def __init__(
        self,
        video_data: List[Dict],
        class_mapping: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        target_frames: int = DATA_CONFIG['num_frames'],
        training: bool = False,
        max_cached_frames: int = 1000  # Limit frame cache size
    ):
        """
        Initialize dataset.
        
        Args:
            video_data: List of video data dictionaries
            class_mapping: Mapping from class names to indices
            transform: Optional transforms to apply
            target_frames: Number of frames to extract
            training: Whether in training mode
            max_cached_frames: Maximum number of frames to cache
        """
        self.video_data = video_data
        self.class_mapping = class_mapping
        self.transform = transform
        self.target_frames = target_frames
        self.training = training
        self.num_classes = len(class_mapping)
        self.max_cached_frames = max_cached_frames
        
        # Default augmentation if no transform provided
        if self.transform is None and training:
            self.transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=DATA_CONFIG['brightness_delta'],
                    contrast=DATA_CONFIG['contrast_range'][0]
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        elif self.transform is None:
            self.transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.video_data)
    
    @lru_cache(maxsize=1000)  # Cache frame loading
    def _load_frame(self, frame_path: str) -> torch.Tensor:
        """
        Load and preprocess single frame with caching.
        
        Args:
            frame_path: Path to frame file
            
        Returns:
            Frame tensor
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, DATA_CONFIG['frame_size'])
        
        # Convert to tensor and normalize
        frame = torch.FloatTensor(frame) / 255.0
        frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        return frame
    
    def load_video(
        self,
        frame_paths: List[str]
    ) -> torch.Tensor:
        """
        Load and preprocess video frames efficiently.
        
        Args:
            frame_paths: List of frame file paths
            
        Returns:
            Frames tensor
        """
        # Calculate frame indices for uniform sampling
        num_frames = len(frame_paths)
        if num_frames >= self.target_frames:
            indices = np.linspace(
                0, num_frames - 1,
                self.target_frames,
                dtype=int
            )
        else:
            indices = list(range(num_frames))
            indices.extend([num_frames - 1] * (self.target_frames - num_frames))
        
        # Load frames with caching
        frames = []
        for idx in indices:
            frame = self._load_frame(frame_paths[idx])
            frames.append(frame)
            
            # Clear cache if too large
            if len(self._load_frame.cache_info().cache) > self.max_cached_frames:
                self._load_frame.cache_clear()
                gc.collect()
        
        return torch.stack(frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        video_info = self.video_data[idx]
        
        # Load frames
        frames = self.load_video(video_info['frame_paths'])
        
        # Apply transforms
        if self.transform:
            frames = torch.stack([
                self.transform(frame) for frame in frames
            ])
        
        # Create one-hot encoded label
        label = torch.zeros(self.num_classes)
        label[self.class_mapping[video_info['gloss']]] = 1
        
        return frames, label

def create_dataloaders(
    video_data: List[Dict],
    class_mapping: Dict[str, int],
    batch_size: int = DATA_CONFIG['batch_size'],
    num_workers: int = 4,
    train_split: float = DATA_CONFIG['train_split'],
    val_split: float = DATA_CONFIG['val_split']
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test dataloaders.
    
    Args:
        video_data: List of video data dictionaries
        class_mapping: Mapping from class names to indices
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Training data proportion
        val_split: Validation data proportion
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    test_split = 1.0 - train_split - val_split
    assert test_split > 0, "Invalid split proportions"
    
    # Shuffle data
    indices = torch.randperm(len(video_data))
    
    # Calculate split indices
    train_idx = int(len(video_data) * train_split)
    val_idx = int(len(video_data) * (train_split + val_split))
    
    # Split data
    train_data = [video_data[i] for i in indices[:train_idx]]
    val_data = [video_data[i] for i in indices[train_idx:val_idx]]
    test_data = [video_data[i] for i in indices[val_idx:]]
    
    # Create datasets with memory efficiency
    max_cached = min(1000, batch_size * num_workers * 2)  # Adjust cache based on workers
    
    train_dataset = VideoDataset(
        train_data,
        class_mapping,
        training=True,
        max_cached_frames=max_cached
    )
    val_dataset = VideoDataset(
        val_data,
        class_mapping,
        training=False,
        max_cached_frames=max_cached
    )
    test_dataset = VideoDataset(
        test_data,
        class_mapping,
        training=False,
        max_cached_frames=max_cached
    )
    
    # Create dataloaders with memory settings
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': 2,  # Reduce prefetching to save memory
        'persistent_workers': True  # Keep workers alive between epochs
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader

def get_class_weights(
    video_data: List[Dict],
    class_mapping: Dict[str, int]
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        video_data: List of video data dictionaries
        class_mapping: Mapping from class names to indices
        
    Returns:
        Tensor of class weights
    """
    # Get class labels
    y = [class_mapping[v['gloss']] for v in video_data]
    
    # Calculate weights
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    
    return torch.FloatTensor(weights)
