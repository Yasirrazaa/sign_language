"""Video preprocessing module for sign language detection."""

import torch
import torchvision.transforms as transforms
import mediapipe as mp
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils import get_video_dir, get_processed_dir
from ..config import DATA_CONFIG

class VideoPreprocessor:
    """Video preprocessing handler."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = DATA_CONFIG['frame_size'],
        device: Optional[torch.device] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target frame size (height, width)
            device: PyTorch device for processing
        """
        self.target_size = target_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
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
    
    def _extract_frames(
        self,
        video_path: Union[str, Path]
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (list of frames, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
        finally:
            cap.release()
        
        return frames, fps
    
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
        
        # Convert back to numpy for saving (scale to 0-255 range)
        frame_np = frame_tensor.permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        return frame_np
    
    def process_video(
        self,
        video_info: Dict
    ) -> Dict:
        """
        Process single video.
        
        Args:
            video_info: Video information dictionary
            
        Returns:
            Processing results dictionary
        """
        try:
            # Get video path
            video_path = get_video_dir() / f"{video_info['video_id']}.mp4"
            
            if not video_path.exists():
                return {
                    'success': False,
                    'error': f"Video file not found: {video_path}",
                    'video_id': video_info['video_id']
                }
            
            # Extract frames
            frames, fps = self._extract_frames(video_path)
            
            if not frames:
                return {
                    'success': False,
                    'error': "No frames extracted from video",
                    'video_id': video_info['video_id']
                }
            
            # Process frames
            processed_frames = []
            bbox = None
            
            for frame in frames:
                # Detect hands if no bbox yet
                if bbox is None:
                    bbox = self._detect_hands(frame)
                
                # Preprocess frame
                processed = self._preprocess_frame(frame)
                processed_frames.append(processed)
            
            # Save frames
            frame_paths = []
            output_dir = get_processed_dir() / 'frames' / video_info['video_id']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(processed_frames):
                frame_path = output_dir / f"frame_{i:04d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )
                frame_paths.append(str(frame_path))
            
            # Include all relevant video info in result
            return {
                'success': True,
                'video_id': video_info['video_id'],
                'gloss': video_info['gloss'],
                'frame_paths': frame_paths,
                'bbox': bbox or video_info.get('bbox', [0, 0, 1, 1]),  # Use provided bbox or default
                'fps': fps,
                'num_frames': len(frames),
                'split': video_info.get('split', 'train'),  # Preserve split information
                'signer_id': video_info.get('signer_id'),  # Preserve signer information
                'instance_id': video_info.get('instance_id')  # Preserve instance information
            }
            
        except Exception as e:
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
        Process batch of videos.
        
        Args:
            video_data: List of video information dictionaries
            num_workers: Number of worker processes
            
        Returns:
            List of processing results
        """
        if not video_data:
            raise ValueError("Empty video data list")
        
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.process_video, video)
                for video in video_data
            ]
            
            for future in tqdm(
                futures,
                total=len(video_data),
                desc="Processing videos"
            ):
                results.append(future.result())
        
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
        # Process videos
        results = self.process_batch(video_data, num_workers)
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        return results
