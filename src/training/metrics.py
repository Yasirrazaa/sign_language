"""Evaluation metrics for sign language detection."""

import torch
import numpy as np
from typing import Dict, Tuple

def calculate_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Calculate classification accuracy and predicted classes.
    
    Args:
        pred: Predicted class probabilities [batch_size, num_classes]
        target: Target class one-hot encodings [batch_size, num_classes]
        
    Returns:
        Tuple of (accuracy, predicted_classes)
    """
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = torch.argmax(target, dim=1)
    correct = (pred_classes == target_classes).float()
    accuracy = correct.mean().item()
    
    return accuracy, pred_classes

def calculate_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor
) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred_boxes: Predicted boxes [batch_size, 4] (x1, y1, x2, y2)
        target_boxes: Target boxes [batch_size, 4] (x1, y1, x2, y2)
        
    Returns:
        Mean IoU score
    """
    # Calculate intersection coordinates
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Calculate areas
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]) *
        (pred_boxes[:, 3] - pred_boxes[:, 1])
    )
    target_area = (
        (target_boxes[:, 2] - target_boxes[:, 0]) *
        (target_boxes[:, 3] - target_boxes[:, 1])
    )
    union = pred_area + target_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou.mean().item()

def calculate_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Calculate confusion matrix.
    
    Args:
        pred: Predicted class probabilities [batch_size, num_classes]
        target: Target class one-hot encodings [batch_size, num_classes]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = torch.argmax(target, dim=1)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        device=pred.device
    )
    
    # Fill confusion matrix
    for t, p in zip(target_classes, pred_classes):
        confusion_matrix[t, p] += 1
    
    return confusion_matrix

def calculate_precision_recall(
    confusion_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate precision and recall for each class.
    
    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        
    Returns:
        Tuple of (precision, recall) tensors
    """
    # Calculate true positives, false positives, false negatives
    true_positives = torch.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    
    return precision, recall

def calculate_f1_score(
    precision: torch.Tensor,
    recall: torch.Tensor
) -> torch.Tensor:
    """
    Calculate F1 score for each class.
    
    Args:
        precision: Precision scores [num_classes]
        recall: Recall scores [num_classes]
        
    Returns:
        F1 scores [num_classes]
    """
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def calculate_metrics(
    class_pred: torch.Tensor,
    class_target: torch.Tensor,
    bbox_pred: torch.Tensor,
    bbox_target: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate all metrics for predictions.
    
    Args:
        class_pred: Predicted class probabilities [batch_size, num_classes]
        class_target: Target class one-hot encodings [batch_size, num_classes]
        bbox_pred: Predicted boxes [batch_size, 4]
        bbox_target: Target boxes [batch_size, 4]
        
    Returns:
        Dictionary of metrics
    """
    # Classification metrics
    accuracy, pred_classes = calculate_accuracy(class_pred, class_target)
    confusion_mat = calculate_confusion_matrix(
        class_pred,
        class_target,
        class_pred.size(1)
    )
    precision, recall = calculate_precision_recall(confusion_mat)
    f1_scores = calculate_f1_score(precision, recall)
    
    # Localization metrics
    iou_score = calculate_iou(bbox_pred, bbox_target)
    
    # Aggregate metrics
    metrics = {
        'accuracy': accuracy,
        'iou': iou_score,
        'mean_precision': precision.mean().item(),
        'mean_recall': recall.mean().item(),
        'mean_f1': f1_scores.mean().item()
    }
    
    return metrics
