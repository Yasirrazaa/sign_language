"""Tests for video preprocessing functionality."""

import unittest
import torch
import numpy as np
import cv2
import json
import shutil
from pathlib import Path
from typing import List

from src.data import VideoPreprocessor
from src.utils import get_video_dir, get_processed_dir
from src.config import DATA_CONFIG

class TestVideoPreprocessor(unittest.TestCase):
    """Test cases for video preprocessing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test directories
        cls.test_dir = Path('test_data')
        cls.test_video_dir = cls.test_dir / 'video'
        cls.test_processed_dir = cls.test_dir / 'processed'
        
        cls.test_video_dir.mkdir(parents=True, exist_ok=True)
        cls.test_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test video
        cls.create_test_video()
        
        # Initialize preprocessor
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.preprocessor = VideoPreprocessor(device=cls.device)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_test_video(cls):
        """Create a test video file."""
        # Create dummy frames
        frames: List[np.ndarray] = []
        for i in range(10):
            # Create frame with a simple shape
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.rectangle(
                frame,
                (20, 20),
                (80, 80),
                (0, 255, 0),
                2
            )
            frames.append(frame)
        
        # Save as video
        video_path = cls.test_video_dir / 'test_video.mp4'
        out = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (100, 100)
        )
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Create video info
        cls.test_video_info = {
            'video_id': 'test_video',
            'gloss': 'test',
            'split': 'train'
        }
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
        self.assertEqual(self.preprocessor.device, self.device)
        self.assertEqual(
            self.preprocessor.target_size,
            DATA_CONFIG['frame_size']
        )
    
    def test_frame_extraction(self):
        """Test video frame extraction."""
        video_path = self.test_video_dir / 'test_video.mp4'
        frames, fps = self.preprocessor._extract_frames(video_path)
        
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], np.ndarray)
        self.assertEqual(len(frames[0].shape), 3)  # H, W, C
        self.assertEqual(frames[0].shape[2], 3)  # RGB
        self.assertGreater(fps, 0)
    
    def test_hand_detection(self):
        """Test hand detection in frames."""
        # Create frame with hand-like shape
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 200, 200), -1)  # Flesh color
        
        bbox = self.preprocessor._detect_hands(frame)
        
        if bbox is not None:  # MediaPipe might not detect our simple shape
            self.assertIsInstance(bbox, list)
            self.assertEqual(len(bbox), 4)  # x1, y1, x2, y2
            self.assertTrue(all(0 <= coord <= 1 for coord in bbox))
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing pipeline."""
        # Create test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (80, 80), (0, 255, 0), 2)
        
        processed = self.preprocessor._preprocess_frame(frame)
        
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape[:2], DATA_CONFIG['frame_size'])
        self.assertEqual(processed.shape[2], 3)  # RGB
        self.assertTrue(np.all((processed >= 0) & (processed <= 1)))
    
    def test_video_processing(self):
        """Test complete video processing pipeline."""
        result = self.preprocessor.process_video(self.test_video_info)
        
        self.assertTrue(result['success'])
        self.assertIn('frame_paths', result)
        self.assertIn('bbox', result)
        self.assertIn('fps', result)
        self.assertIn('num_frames', result)
        
        # Check frame paths
        self.assertIsInstance(result['frame_paths'], list)
        self.assertGreater(len(result['frame_paths']), 0)
        for path in result['frame_paths']:
            self.assertTrue(Path(path).exists())
    
    def test_batch_processing(self):
        """Test batch video processing."""
        batch = [self.test_video_info] * 2  # Process same video twice
        results = self.preprocessor.process_batch(batch, num_workers=1)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result['success'])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid video path
        invalid_info = {
            'video_id': 'nonexistent',
            'gloss': 'test',
            'split': 'train'
        }
        result = self.preprocessor.process_video(invalid_info)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
        # Test empty batch
        with self.assertRaises(ValueError):
            self.preprocessor.process_batch([])
    
    def test_results_saving(self):
        """Test saving processing results."""
        results = [
            {
                'success': True,
                'frame_paths': ['test/path1.jpg', 'test/path2.jpg'],
                'bbox': [0.1, 0.2, 0.3, 0.4],
                'fps': 30,
                'num_frames': 2
            }
        ]
        
        output_path = self.test_processed_dir / 'test_results.json'
        self.preprocessor._save_results(results, output_path)
        
        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, results)

def main():
    """Run the tests."""
    unittest.main()

if __name__ == '__main__':
    main()
