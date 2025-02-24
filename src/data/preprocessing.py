"""Video preprocessing module for sign language detection."""

import torch
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np
import cv2
import json
import gc
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Generator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils import get_video_dir, get_processed_dir
from ..config import DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Memory-efficient video preprocessing handler."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = DATA_CONFIG['frame_size'],
        device: Optional[torch.device] = None,
        max_frames: int = 300,  # Limit number of frames
        sampling_rate: int = 2  # Sample every nth frame
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target frame size (height, width)
            device: PyTorch device for processing
            max_frames: Maximum number of frames to process per video
            sampling_rate: Process every nth frame
        """
        self.target_size = target_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.max_frames = max_frames
        self.sampling_rate = sampling_rate
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Initialize transform pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _frame_generator(
        self,
        video_path: Union[str, Path]
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generate frames from video file.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while frame_count < min(total_frames, self.max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % self.sampling_rate == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame, fps
                
                frame_count += 1
                
        finally:
            cap.release()

    def _detect_hands(
        self,
        frame: np.ndarray
    ) -> Optional[List[float]]:
        """
        Detect hands in frame and return bounding box.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized bounding box coordinates [x1, y1, x2, y2] or None
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Process frame
        results = self.mp_hands.process(frame)
        
        if results.multi_hand_landmarks:
            # Get all hand landmarks
            all_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [
                    (lm.x * width, lm.y * height)
                    for lm in hand_landmarks.landmark
                ]
                all_landmarks.extend(landmarks)
            
            # Calculate bounding box
            x_coords = [x for x, y in all_landmarks]
            y_coords = [y for x, y in all_landmarks]
            
            x1 = max(0, min(x_coords))
            y1 = max(0, min(y_coords))
            x2 = min(width, max(x_coords))
            y2 = min(height, max(y_coords))
            
            # Add margin
            margin = 0.1
            w = x2 - x1
            h = y2 - y1
            x1 = max(0, x1 - w * margin)
            y1 = max(0, y1 - h * margin)
            x2 = min(width, x2 + w * margin)
            y2 = min(height, y2 + h * margin)
            
            # Normalize coordinates
            return [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ]
        
        return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to tensor
        frame_tensor = self.transform(frame)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = frame_tensor * std + mean
        
        # Convert back to numpy for saving
        frame_np = frame_tensor.permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        del frame_tensor
        return frame_np

    def process_video(
        self,
        video_info: Dict
    ) -> Dict:
        """
        Process single video with memory efficiency.
        
        Args:
            video_info: Video information dictionary
            
        Returns:
            Processing results dictionary
        """
        try:
            video_path = get_video_dir() / f"{video_info['video_id']}.mp4"
            
            if not video_path.exists():
                return {
                    'success': False,
                    'error': f"Video file not found: {video_path}",
                    'video_id': video_info['video_id']
                }
            
            # Setup output directory
            output_dir = get_processed_dir() / 'frames' / video_info['video_id']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            frame_paths = []
            frame_count = 0
            bbox = None
            fps = None
            
            # Process frames as they are generated
            for frame, current_fps in self._frame_generator(video_path):
                if fps is None:
                    fps = current_fps
                
                # Detect hands if no bbox yet
                if bbox is None:
                    bbox = self._detect_hands(frame)
                
                # Preprocess frame
                processed = self._preprocess_frame(frame)
                
                # Save frame immediately
                frame_path = output_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90]  # Reduced quality for memory efficiency
                )
                
                frame_paths.append(str(frame_path))
                frame_count += 1
                
                # Clear memory
                del processed
                if frame_count % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if frame_count == 0:
                return {
                    'success': False,
                    'error': "No frames processed from video",
                    'video_id': video_info['video_id']
                }
            
            return {
                'success': True,
                'video_id': video_info['video_id'],
                'gloss': video_info['gloss'],
                'frame_paths': frame_paths,
                'bbox': bbox or video_info.get('bbox', [0, 0, 1, 1]),
                'fps': fps,
                'num_frames': frame_count,
                'split': video_info.get('split', 'train'),
                'signer_id': video_info.get('signer_id'),
                'instance_id': video_info.get('instance_id')
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_info.get('video_id', 'unknown')}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'video_id': video_info.get('video_id', 'unknown')
            }

    def process_batch(
        self,
        video_data: List[Dict],
        num_workers: int = 4
    ) -> List[Dict]:
        """
        Process batch of videos with memory monitoring.
        
        Args:
            video_data: List of video information dictionaries
            num_workers: Maximum number of worker processes
            
        Returns:
            List of processing results
        """
        if not video_data:
            raise ValueError("Empty video data list")
        
        results = []
        total_videos = len(video_data)
        processed = 0
        
        # Get baseline memory usage
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
        logger.info(f"Initial memory usage: {memory_usage:.2f} GB")
        
        while processed < total_videos:
            # Check available memory
            available_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            batch_size = max(1, min(int(available_mem / 2), 8))  # Use at most half available memory
            current_batch = video_data[processed:processed + batch_size]
            
            logger.info(f"Processing batch of {len(current_batch)} videos")
            
            with ThreadPoolExecutor(max_workers=min(num_workers, batch_size)) as executor:
                futures = [
                    executor.submit(self.process_video, video)
                    for video in current_batch
                ]
                
                for future in tqdm(futures, desc=f"Batch {processed//batch_size + 1}"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {str(e)}")
            
            processed += len(current_batch)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory usage
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            logger.info(f"Memory usage after batch: {memory_usage:.2f} GB")
        
        return results

    def _save_results(
        self,
        results: List[Dict],
        output_path: Optional[Path] = None
    ):
        """
        Save preprocessing results.
        
        Args:
            results: List of processing results
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = get_processed_dir() / 'preprocessing_results.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def __call__(
        self,
        video_data: List[Dict],
        num_workers: int = 4,
        save_results: bool = True
    ) -> List[Dict]:
        """
        Process videos and optionally save results.
        
        Args:
            video_data: List of video information dictionaries
            num_workers: Number of worker processes
            save_results: Whether to save results to disk
            
        Returns:
            List of processing results
        """
        results = self.process_batch(video_data, num_workers)
        
        if save_results:
            self._save_results(results)
        
        return results
