"""Evaluation metrics for sign language detection."""

import torch
import numpy as np
from typing import Dict, Tuple

def calculate_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        pred: Predicted class probabilities [batch_size, num_classes]
        target: Target class one-hot encodings [batch_size, num_classes]
        
    Returns:
        Accuracy score
    """
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = torch.argmax(target, dim=1)
    correct = (pred_classes == target_classes).float()
    accuracy = correct.mean().item()
    
    return accuracy

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
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate all metrics for predictions.
    
    Args:
        pred: Predicted class probabilities [batch_size, num_classes]
        target: Target class one-hot encodings [batch_size, num_classes]
        
    Returns:
        Dictionary of metrics
    """
    # Calculate accuracy
    accuracy = calculate_accuracy(pred, target)
    
    # Calculate confusion matrix
    confusion_mat = calculate_confusion_matrix(
        pred,
        target,
        pred.size(1)
    )
    
    # Calculate precision and recall
    precision, recall = calculate_precision_recall(confusion_mat)
    
    # Calculate F1 score
    f1_scores = calculate_f1_score(precision, recall)
    
    # Aggregate metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1_scores.mean().item()
    }
    
    return metrics
