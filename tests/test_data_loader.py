"""Tests for data loading functionality."""

import unittest
import torch
import numpy as np
from pathlib import Path

from src.data import (
    VideoDataset,
    create_dataloaders,
    load_video_data,
    get_class_weights
)
from src.config import DATA_CONFIG

class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Sample video data
        cls.video_data = [
            {
                'video_id': 'test_video_1',
                'frame_paths': [
                    f'test_frames/test_video_1/frame_{i:04d}.jpg'
                    for i in range(10)
                ],
                'bbox': [0.1, 0.2, 0.3, 0.4],
                'fps': 30,
                'num_frames': 10,
                'gloss': 'hello'
            },
            {
                'video_id': 'test_video_2',
                'frame_paths': [
                    f'test_frames/test_video_2/frame_{i:04d}.jpg'
                    for i in range(15)
                ],
                'bbox': [0.2, 0.3, 0.4, 0.5],
                'fps': 30,
                'num_frames': 15,
                'gloss': 'thank_you'
            }
        ]
        
        # Sample class mapping
        cls.class_mapping = {'hello': 0, 'thank_you': 1}
        
        # Create test device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = VideoDataset(
            self.video_data,
            self.class_mapping
        )
        
        self.assertEqual(len(dataset), len(self.video_data))
        self.assertEqual(dataset.num_classes, len(self.class_mapping))
        self.assertIsNotNone(dataset.transform)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = VideoDataset(
            self.video_data,
            self.class_mapping,
            target_frames=5  # Small number for testing
        )
        
        # Mock frame loading
        def mock_load_video(*args):
            frames = torch.randn(5, 3, *DATA_CONFIG['frame_size'])
            bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
            return frames, bbox
        
        # Replace actual loading with mock
        dataset.load_video = mock_load_video
        
        # Get an item
        frames, (label, bbox) = dataset[0]
        
        # Check shapes
        self.assertEqual(
            frames.shape,
            (5, 3, *DATA_CONFIG['frame_size'])
        )
        self.assertEqual(label.shape, (len(self.class_mapping),))
        self.assertEqual(bbox.shape, (4,))
        
        # Check types
        self.assertTrue(torch.is_tensor(frames))
        self.assertTrue(torch.is_tensor(label))
        self.assertTrue(torch.is_tensor(bbox))
        
        # Check values
        self.assertTrue(torch.all(label >= 0) and torch.all(label <= 1))
        self.assertTrue(torch.all(bbox >= 0) and torch.all(bbox <= 1))
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.video_data,
            self.class_mapping,
            batch_size=2
        )
        
        # Check loader types
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Check batch size
        self.assertEqual(train_loader.batch_size, 2)
        self.assertEqual(val_loader.batch_size, 2)
        self.assertEqual(test_loader.batch_size, 2)
    
    def test_class_weights(self):
        """Test class weight calculation."""
        weights = get_class_weights(
            self.video_data,
            self.class_mapping
        )
        
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), len(self.class_mapping))
        self.assertTrue(torch.all(weights > 0))
    
    def test_data_loading_errors(self):
        """Test error handling in data loading."""
        # Test invalid class
        with self.assertRaises(KeyError):
            dataset = VideoDataset(
                [{'gloss': 'invalid_class'}],
                self.class_mapping
            )
            _ = dataset[0]
        
        # Test invalid split proportions
        with self.assertRaises(AssertionError):
            _ = create_dataloaders(
                self.video_data,
                self.class_mapping,
                train_split=0.9,
                val_split=0.2  # Makes total > 1
            )
    
    def test_transform_application(self):
        """Test data augmentation transforms."""
        dataset = VideoDataset(
            self.video_data,
            self.class_mapping,
            training=True
        )
        
        # Mock frame loading
        def mock_load_video(*args):
            frames = torch.ones(5, 3, *DATA_CONFIG['frame_size'])
            bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
            return frames, bbox
        
        dataset.load_video = mock_load_video
        
        # Get transformed item
        frames, _ = dataset[0]
        
        # Check if transforms were applied
        self.assertTrue(
            not torch.all(frames == 1.0),
            "Transforms should modify pixel values"
        )

def main():
    """Run the tests."""
    unittest.main()

if __name__ == '__main__':
    main()
