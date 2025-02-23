"""Visualization utilities for sign language detection."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

class VideoVisualizer:
    """Video visualization handler."""
    
    def __init__(
        self,
        class_mapping: Dict[str, int],
        figsize: Tuple[int, int] = (12, 8),
        font_size: int = 12
    ):
        """
        Initialize visualizer.
        
        Args:
            class_mapping: Class name to index mapping
            figsize: Figure size for plots
            font_size: Font size for text
        """
        self.class_mapping = class_mapping
        self.figsize = figsize
        self.font_size = font_size
        
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # Set color map for classes
        self.colors = plt.cm.rainbow(
            np.linspace(0, 1, len(class_mapping))
        )
    
    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: List[float],
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        color: Optional[Tuple[float, ...]] = None
    ) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: Normalized coordinates [x1, y1, x2, y2]
            label: Optional class label
            confidence: Optional prediction confidence
            color: Optional box color (R, G, B, A)
            
        Returns:
            Frame with drawn bbox
        """
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixels
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # Set default color
        if color is None:
            color = (0, 255, 0, 255)  # Green
        
        # Convert to uint8 RGB
        color_rgb = tuple(int(c * 255) for c in color[:3])
        
        # Draw bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color_rgb,
            2
        )
        
        # Add label if provided
        if label:
            label_text = label
            if confidence is not None:
                label_text += f" ({confidence:.2f})"
            
            # Draw label background
            text_size = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )[0]
            cv2.rectangle(
                frame,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                color_rgb,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def visualize_prediction(
        self,
        frame: np.ndarray,
        class_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize model predictions on frame.
        
        Args:
            frame: Input frame
            class_pred: Class prediction logits
            bbox_pred: Bounding box prediction
            num_classes: Optional number of top classes to show
            
        Returns:
            Annotated frame
        """
        # Get class probabilities
        probs = torch.softmax(class_pred, dim=0)
        
        # Get top predictions
        if num_classes is None:
            num_classes = len(self.class_mapping)
        
        top_probs, top_indices = torch.topk(probs, num_classes)
        
        # Draw top predictions
        frame_copy = frame.copy()
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = self.idx_to_class[idx.item()]
            color = self.colors[idx]
            
            if i == 0:  # Draw bbox only for top prediction
                frame_copy = self.draw_bbox(
                    frame_copy,
                    bbox_pred.tolist(),
                    label,
                    prob.item(),
                    color
                )
            
            # Add prediction to legend
            y_pos = 30 + i * 20
            cv2.putText(
                frame_copy,
                f"{label}: {prob.item():.2f}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                tuple(int(c * 255) for c in color[:3]),
                1,
                cv2.LINE_AA
            )
        
        return frame_copy
    
    def create_grid(
        self,
        frames: List[np.ndarray],
        predictions: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_cols: int = 4
    ) -> np.ndarray:
        """
        Create grid of frames with predictions.
        
        Args:
            frames: List of input frames
            predictions: Optional list of (class_pred, bbox_pred) pairs
            num_cols: Number of columns in grid
            
        Returns:
            Grid image
        """
        num_frames = len(frames)
        num_rows = (num_frames + num_cols - 1) // num_cols
        
        # Create empty grid
        cell_height, cell_width = frames[0].shape[:2]
        grid = np.zeros(
            (cell_height * num_rows, cell_width * num_cols, 3),
            dtype=np.uint8
        )
        
        # Fill grid
        for i, frame in enumerate(frames):
            row = i // num_cols
            col = i % num_cols
            
            # Add predictions if available
            if predictions is not None and i < len(predictions):
                class_pred, bbox_pred = predictions[i]
                frame = self.visualize_prediction(frame, class_pred, bbox_pred)
            
            # Add to grid
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width
            grid[y1:y2, x1:x2] = frame
        
        return grid
    
    def save_visualization(
        self,
        image: np.ndarray,
        output_path: Union[str, Path]
    ):
        """
        Save visualization to file.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

def plot_predictions(
    frames: List[np.ndarray],
    predictions: List[Tuple[torch.Tensor, torch.Tensor]],
    class_mapping: Dict[str, int],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot frames with predictions.
    
    Args:
        frames: List of input frames
        predictions: List of (class_pred, bbox_pred) pairs
        class_mapping: Class name to index mapping
        output_path: Optional path to save plot
        show: Whether to display plot
    """
    visualizer = VideoVisualizer(class_mapping)
    grid = visualizer.create_grid(frames, predictions)
    
    if output_path is not None:
        visualizer.save_visualization(grid, output_path)
    
    if show:
        plt.figure(figsize=(15, 15))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()
