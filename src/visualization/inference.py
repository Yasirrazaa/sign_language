"""Real-time inference utilities for sign language detection."""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List,Dict, Optional, Tuple, Union
from queue import Queue
from threading import Thread
import time

from ..models import SignLanguageCNNLSTM, VideoTransformer
from .visualizer import VideoVisualizer
from ..data import VideoPreprocessor

class RealTimeInference:
    """Real-time video inference handler."""
    
    def __init__(
        self,
        model: Union[SignLanguageCNNLSTM, VideoTransformer],
        class_mapping: Dict[str, int],
        frame_buffer_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference handler.
        
        Args:
            model: Trained model
            class_mapping: Class name to index mapping
            frame_buffer_size: Size of frame buffer for prediction
            device: Device to run inference on
        """
        self.model = model
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor()
        
        # Initialize visualizer
        self.visualizer = VideoVisualizer(class_mapping)
        
        # Create frame queue for async processing
        self.frame_queue = Queue(maxsize=frame_buffer_size)
        self.result_queue = Queue(maxsize=1)
        self.running = False
    
    def preprocess_frame(
        self,
        frame: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame tensor
        """
        # Preprocess frame
        frame = self.preprocessor._preprocess_frame(frame)
        
        # Convert to tensor
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)  # HWC -> CHW
        
        return frame
    
    @torch.no_grad()
    def process_frames(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            Tuple of (class_predictions, bbox_predictions)
        """
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)
        
        # Stack frames
        frame_tensor = torch.stack(processed_frames)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        frame_tensor = frame_tensor.to(self.device)
        
        # Get predictions
        class_pred, bbox_pred = self.model(frame_tensor)
        
        return class_pred[0], bbox_pred[0]  # Remove batch dimension
    
    def inference_worker(self):
        """Background worker for inference."""
        while self.running:
            # Get frames from queue
            if len(self.frame_buffer) < self.frame_buffer_size:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.frame_buffer.append(frame)
                continue
            
            # Process frames
            class_pred, bbox_pred = self.process_frames(self.frame_buffer)
            
            # Update result queue
            if not self.result_queue.full():
                self.result_queue.put((class_pred, bbox_pred))
            
            # Update frame buffer
            self.frame_buffer = self.frame_buffer[1:]
    
    def start(
        self,
        source: Union[int, str, Path] = 0,
        output_path: Optional[Union[str, Path]] = None,
        window_name: str = "Sign Language Detection"
    ):
        """
        Start real-time inference.
        
        Args:
            source: Video source (0 for webcam)
            output_path: Optional path to save output video
            window_name: Name of display window
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video source")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if needed
        writer = None
        if output_path is not None:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        # Start inference thread
        self.running = True
        inference_thread = Thread(target=self.inference_worker)
        inference_thread.start()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add to frame queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_rgb)
                
                # Get latest prediction
                if not self.result_queue.empty():
                    class_pred, bbox_pred = self.result_queue.get()
                    frame = self.visualizer.visualize_prediction(
                        frame_rgb,
                        class_pred,
                        bbox_pred
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Save frame if needed
                if writer is not None:
                    writer.write(frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Cleanup
            self.running = False
            inference_thread.join()
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ):
        """
        Process video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
        """
        self.start(
            source=str(video_path),
            output_path=output_path
        )
