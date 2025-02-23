"""Memory-efficient video processing for sign language recognition."""

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm

from ...configs.base_config import (
    DATA_CONFIG,
    PREPROCESSING_CONFIG,
    PROCESSED_VIDEOS_DIR,
    FRAMES_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Memory-efficient video processing class."""
    
    def __init__(self, 
                 frame_size: Tuple[int, int] = DATA_CONFIG['frame_size'],
                 num_frames: int = DATA_CONFIG['num_frames'],
                 fps: int = DATA_CONFIG['fps']):
        """Initialize video processor with configuration."""
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.fps = fps
        self.chunk_size = PREPROCESSING_CONFIG['chunk_size']
        
    def process_video(self, 
                     video_path: Path,
                     output_dir: Optional[Path] = None,
                     start_frame: int = 0,
                     end_frame: Optional[int] = None) -> np.ndarray:
        """
        Process video in memory-efficient chunks.
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save processed frames
            start_frame: Starting frame number
            end_frame: Ending frame number (None for entire video)
            
        Returns:
            Processed frames as numpy array
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
            
        # Validate frame range
        if start_frame < 0 or end_frame > total_frames:
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}")
            
        # Calculate target frames based on FPS
        target_frames = self._get_target_frames(cap, start_frame, end_frame)
        
        # Process in chunks to manage memory
        frames = []
        for chunk_start in range(0, len(target_frames), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(target_frames))
            chunk_frames = self._process_frame_chunk(cap, 
                                                   target_frames[chunk_start:chunk_end],
                                                   output_dir)
            frames.extend(chunk_frames)
            
        cap.release()
        
        # Convert to numpy array and normalize
        frames = np.array(frames)
        frames = frames.astype(np.float32) / 255.0
        
        return frames
    
    def _get_target_frames(self, 
                          cap: cv2.VideoCapture,
                          start_frame: int,
                          end_frame: int) -> List[int]:
        """Calculate target frame indices based on desired FPS."""
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps))
        
        target_frames = list(range(start_frame, end_frame, frame_interval))
        
        # Interpolate or sample frames if needed
        if len(target_frames) > self.num_frames:
            # Sample frames uniformly
            indices = np.linspace(0, len(target_frames)-1, self.num_frames, dtype=int)
            target_frames = [target_frames[i] for i in indices]
        elif len(target_frames) < self.num_frames:
            # Use duplicate frames
            target_frames = target_frames + [target_frames[-1]] * (self.num_frames - len(target_frames))
            
        return target_frames[:self.num_frames]
    
    def _process_frame_chunk(self,
                           cap: cv2.VideoCapture,
                           frame_indices: List[int],
                           output_dir: Optional[Path]) -> List[np.ndarray]:
        """Process a chunk of frames."""
        processed_frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")
                
            # Resize frame
            frame = cv2.resize(frame, self.frame_size)
            
            # Save frame if output directory provided
            if output_dir:
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame, 
                          [cv2.IMWRITE_JPEG_QUALITY, PREPROCESSING_CONFIG['compression_quality']])
            
            processed_frames.append(frame)
            
        return processed_frames

    @staticmethod
    def clean_tmp_files():
        """Clean temporary files to free disk space."""
        if PREPROCESSING_CONFIG['cleanup_tmp']:
            tmp_dir = PREPROCESSING_CONFIG['tmp_dir']
            if tmp_dir.exists():
                for file in tmp_dir.glob("*"):
                    file.unlink()
                logging.info(f"Cleaned temporary files in {tmp_dir}")

class BatchVideoProcessor:
    """Process multiple videos in batches."""
    
    def __init__(self, video_processor: VideoProcessor):
        """Initialize with video processor instance."""
        self.video_processor = video_processor
        
    def process_batch(self, 
                     video_paths: List[Path],
                     output_base_dir: Optional[Path] = None) -> None:
        """
        Process multiple videos in memory-efficient batches.
        
        Args:
            video_paths: List of video paths to process
            output_base_dir: Optional base directory for saving processed frames
        """
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                if output_base_dir:
                    output_dir = output_base_dir / video_path.stem
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = None
                    
                self.video_processor.process_video(video_path, output_dir)
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                continue
            
        # Clean up temporary files
        VideoProcessor.clean_tmp_files()