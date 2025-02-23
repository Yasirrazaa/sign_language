"""Memory-efficient preprocessing for sign language videos."""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientPreprocessor:
    """Memory-efficient video preprocessing with batch processing and streaming."""
    
    def __init__(self, 
                 output_dir: Path,
                 frame_size: Tuple[int, int] = (224, 224),
                 target_fps: int = 25,
                 chunk_size: int = 32,
                 max_frames: int = 64,
                 num_workers: int = 4):
        """
        Initialize preprocessor with memory-efficient settings.
        
        Args:
            output_dir: Directory to save processed frames
            frame_size: Target frame size (height, width)
            target_fps: Target frames per second
            chunk_size: Number of frames to process at once
            max_frames: Maximum frames to keep per video
            num_workers: Number of worker threads
        """
        self.output_dir = Path(output_dir)
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.chunk_size = chunk_size
        self.max_frames = max_frames
        self.num_workers = num_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def preprocess_video(self, 
                        video_path: Path,
                        start_frame: Optional[int] = None,
                        end_frame: Optional[int] = None) -> Dict:
        """
        Preprocess a single video with memory efficiency.
        
        Args:
            video_path: Path to video file
            start_frame: Optional starting frame
            end_frame: Optional ending frame
            
        Returns:
            Dictionary with preprocessing results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices
        frame_indices = self._get_frame_indices(
            fps, total_frames, start_frame, end_frame
        )
        
        # Process frames in chunks
        processed_frames = []
        output_frames = []
        
        for i in range(0, len(frame_indices), self.chunk_size):
            chunk_indices = frame_indices[i:i + self.chunk_size]
            
            # Process chunk
            chunk_frames = self._process_frame_chunk(cap, chunk_indices)
            processed_frames.extend(chunk_frames)
            
            # Save frames if we have enough
            if len(processed_frames) >= self.chunk_size:
                self._save_frames(video_path.stem, processed_frames[:self.chunk_size])
                output_frames.extend(processed_frames[:self.chunk_size])
                processed_frames = processed_frames[self.chunk_size:]
            
            # Clear memory
            gc.collect()
        
        # Save any remaining frames
        if processed_frames:
            self._save_frames(video_path.stem, processed_frames)
            output_frames.extend(processed_frames)
        
        cap.release()
        
        return {
            'video_id': video_path.stem,
            'num_frames': len(output_frames),
            'frame_size': self.frame_size,
            'fps': self.target_fps
        }
    
    def batch_preprocess(self, video_paths: List[Path]) -> List[Dict]:
        """
        Preprocess multiple videos in parallel with memory efficiency.
        
        Args:
            video_paths: List of video paths to process
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for video_path in video_paths:
                future = executor.submit(self.preprocess_video, video_path)
                futures.append(future)
            
            # Process results as they complete
            for future in tqdm(futures, desc="Processing videos"):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Clear memory periodically
                    if len(results) % 10 == 0:
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error processing video: {str(e)}")
        
        return results
    
    def _get_frame_indices(self,
                          original_fps: float,
                          total_frames: int,
                          start_frame: Optional[int],
                          end_frame: Optional[int]) -> List[int]:
        """Calculate frame indices to extract."""
        # Handle start/end frames
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else total_frames
        
        # Calculate frame step to achieve target FPS
        step = max(1, int(original_fps / self.target_fps))
        
        # Get frame indices
        indices = list(range(start, end, step))
        
        # Limit number of frames if needed
        if len(indices) > self.max_frames:
            # Sample frames uniformly
            indices = np.linspace(0, len(indices)-1, self.max_frames, dtype=int)
            indices = [indices[i] for i in indices]
        
        return indices
    
    def _process_frame_chunk(self,
                           cap: cv2.VideoCapture,
                           frame_indices: List[int]) -> List[np.ndarray]:
        """Process a chunk of frames."""
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        
        return frames
    
    def _save_frames(self, video_id: str, frames: List[np.ndarray]):
        """Save processed frames to disk."""
        video_dir = self.output_dir / video_id
        video_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = video_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

def main():
    """Run preprocessing on video dataset."""
    try:
        # Set up paths
        video_dir = Path("video")
        processed_dir = Path("processed/frames")
        
        # Get video paths
        video_paths = list(video_dir.glob("**/*.mp4"))
        
        if not video_paths:
            logger.error("No videos found for preprocessing")
            return
            
        logger.info(f"Found {len(video_paths)} videos to process")
        
        # Initialize preprocessor
        preprocessor = MemoryEfficientPreprocessor(
            output_dir=processed_dir,
            frame_size=(224, 224),
            target_fps=25,
            chunk_size=32
        )
        
        # Process videos
        results = preprocessor.batch_preprocess(video_paths)
        
        # Save preprocessing results
        with open("processed/preprocessing_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
