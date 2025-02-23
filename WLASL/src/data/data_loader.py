"""Memory-efficient data loading with cross-validation support."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from sklearn.model_selection import KFold
import logging
from tqdm import tqdm

from ...configs.base_config import DATA_CONFIG, FRAMES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """Memory efficient dataset for sign language videos."""
    
    def __init__(self,
                 data_info: List[Dict],
                 transform=None,
                 mode: str = 'train',
                 fold_idx: Optional[int] = None,
                 num_folds: int = DATA_CONFIG['num_folds']):
        """
        Initialize dataset.
        
        Args:
            data_info: List of dictionaries containing video metadata
            transform: Optional transforms to apply
            mode: 'train', 'val', or 'test'
            fold_idx: Current fold index for cross-validation
            num_folds: Total number of folds
        """
        self.data_info = data_info
        self.transform = transform
        self.mode = mode
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        
        # Set up cross-validation splits if needed
        if fold_idx is not None:
            self.setup_cross_validation()
        
        # Memory efficient frame loading
        self.frame_cache = {}
        self.max_cache_size = 100  # Adjust based on available memory
        
    def setup_cross_validation(self):
        """Setup data splits for cross-validation."""
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        
        # Get indices for current fold
        all_indices = np.arange(len(self.data_info))
        splits = list(kf.split(all_indices))
        train_idx, val_idx = splits[self.fold_idx]
        
        if self.mode == 'train':
            self.indices = train_idx
        elif self.mode == 'val':
            self.indices = val_idx
        else:  # test mode uses all data
            self.indices = all_indices
            
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        if hasattr(self, 'indices'):
            return len(self.indices)
        return len(self.data_info)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        if hasattr(self, 'indices'):
            idx = self.indices[idx]
            
        sample_info = self.data_info[idx]
        
        # Load frames efficiently
        frames = self._load_frames(sample_info)
        
        if self.transform:
            frames = self.transform(frames)
            
        # Convert to tensor and normalize
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W) format for I3D
        
        label = sample_info['label']
        
        return frames, label
    
    def _load_frames(self, sample_info: Dict) -> np.ndarray:
        """Load frames with caching for memory efficiency."""
        video_id = sample_info['video_id']
        
        # Try to get from cache first
        if video_id in self.frame_cache:
            return self.frame_cache[video_id]
            
        # Load frames from disk
        frame_dir = FRAMES_DIR / video_id
        frame_paths = sorted(frame_dir.glob('*.jpg'))
        
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        frames = np.array(frames)
        
        # Update cache
        if len(self.frame_cache) >= self.max_cache_size:
            # Remove oldest item if cache is full
            self.frame_cache.pop(next(iter(self.frame_cache)))
            
        self.frame_cache[video_id] = frames
        
        return frames

def create_data_loaders(data_info: List[Dict],
                       batch_size: int = DATA_CONFIG['batch_size'],
                       num_workers: int = DATA_CONFIG['num_workers'],
                       fold_idx: Optional[int] = None) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_info: List of dictionaries containing video metadata
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        fold_idx: Current fold index for cross-validation
        
    Returns:
        Dictionary containing data loaders for each split
    """
    # Create datasets
    train_dataset = SignLanguageDataset(
        data_info,
        mode='train',
        fold_idx=fold_idx
    )
    
    val_dataset = SignLanguageDataset(
        data_info,
        mode='val',
        fold_idx=fold_idx
    )
    
    test_dataset = SignLanguageDataset(
        data_info,
        mode='test',
        fold_idx=fold_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DATA_CONFIG['pin_memory'],
        prefetch_factor=DATA_CONFIG['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }