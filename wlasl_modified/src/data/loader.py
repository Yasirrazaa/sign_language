"""Memory-efficient data loading for sign language recognition."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameCache:
    """LRU cache for video frames with memory management."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize frame cache.
        
        Args:
            max_size: Maximum number of frames to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.usage = defaultdict(int)
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get frame from cache and update usage."""
        if key in self.cache:
            self.usage[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, frame: np.ndarray):
        """Add frame to cache with memory management."""
        # Check if we need to free up space
        if len(self.cache) >= self.max_size:
            # Remove least used items
            items = sorted(self.usage.items(), key=lambda x: x[1])
            to_remove = items[:len(items)//4]  # Remove 25% of least used items
            
            for k, _ in to_remove:
                if k in self.cache:
                    del self.cache[k]
                del self.usage[k]
        
        self.cache[key] = frame
        self.usage[key] = 1

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset for sign language videos."""
    
    def __init__(self,
                 data: List[Dict],
                 processed_dir: Path,
                 frame_cache_size: int = 1000,
                 transform=None):
        """
        Initialize dataset with memory efficiency.
        
        Args:
            data: List of video metadata
            processed_dir: Directory containing processed frames
            frame_cache_size: Size of frame cache
            transform: Optional transforms to apply
        """
        self.data = data
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.frame_cache = FrameCache(frame_cache_size)
        
        # Precompute frame paths for efficiency
        self.frame_paths = {}
        for item in data:
            video_id = item['video_id']
            video_dir = self.processed_dir / video_id
            if video_dir.exists():
                self.frame_paths[video_id] = sorted(list(video_dir.glob("*.jpg")))
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset with memory efficiency.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        item = self.data[idx]
        video_id = item['video_id']
        
        # Load frames efficiently
        frames = []
        frame_paths = self.frame_paths.get(video_id, [])
        
        for frame_path in frame_paths:
            frame_key = f"{video_id}_{frame_path.name}"
            
            # Try to get frame from cache
            frame = self.frame_cache.get(frame_key)
            
            if frame is None:
                # Load frame and add to cache
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_cache.put(frame_key, frame)
            
            frames.append(frame)
        
        # Convert to tensor
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        frames = frames / 255.0  # Normalize
        
        if self.transform:
            frames = self.transform(frames)
        
        # Get label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return frames, label

def create_data_loaders(
    data_info: List[Dict],
    processed_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    frame_cache_size: int = 1000
) -> Dict[str, DataLoader]:
    """
    Create memory-efficient data loaders.
    
    Args:
        data_info: List of video metadata
        processed_dir: Directory containing processed frames
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        frame_cache_size: Size of frame cache per worker
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Split data by set
    split_data = defaultdict(list)
    for item in data_info:
        split_data[item['split']].append(item)
    
    # Create datasets
    datasets = {
        split: MemoryEfficientDataset(
            data=items,
            processed_dir=processed_dir,
            frame_cache_size=frame_cache_size
        )
        for split, items in split_data.items()
    }
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders

class StreamingDataLoader:
    """Memory-efficient data loader for large datasets."""
    
    def __init__(self,
                 data: List[Dict],
                 processed_dir: Path,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize streaming data loader.
        
        Args:
            data: List of video metadata
            processed_dir: Directory containing processed frames
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        
        if shuffle:
            self.indices = np.random.permutation(len(data))
        else:
            self.indices = np.arange(len(data))
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self):
        """Get next batch."""
        if self.current_idx >= len(self.data):
            self.current_idx = 0
            if self.shuffle:
                self.indices = np.random.permutation(len(self.data))
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Load batch data
        frames_list = []
        labels_list = []
        
        for idx in batch_indices:
            item = self.data[idx]
            video_id = item['video_id']
            
            # Load frames
            video_dir = self.processed_dir / video_id
            frame_paths = sorted(list(video_dir.glob("*.jpg")))
            
            frames = []
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            # Convert to tensor
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            frames = frames / 255.0
            
            frames_list.append(frames)
            labels_list.append(item['label'])
        
        # Stack batch
        batch_frames = torch.stack(frames_list)
        batch_labels = torch.tensor(labels_list, dtype=torch.long)
        
        return batch_frames, batch_labels

def load_data_info(json_path: Path) -> List[Dict]:
    """Load data info from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    # Example usage
    data_info = load_data_info(Path('processed/preprocessing_results.json'))
    dataloaders = create_data_loaders(data_info, Path('processed/frames'))
    
    # Print dataset sizes
    for split, loader in dataloaders.items():
        print(f"{split} dataset size: {len(loader.dataset)}")
