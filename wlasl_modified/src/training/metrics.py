"""Evaluation metrics for sign language detection."""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support

def calculate_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        pred: Predicted logits [batch_size, num_classes]
        target: Target class indices [batch_size]
        
    Returns:
        Accuracy score
    """
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target).float()
    accuracy = correct.mean().item()
    
    return accuracy

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate all metrics for predictions.
    
    Args:
        predictions: Predicted logits [batch_size, num_classes]
        targets: Target class indices [batch_size]
        
    Returns:
        Dictionary of metrics
    """
    # Move tensors to CPU and convert to numpy
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, targets)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np,
        pred_classes,
        average='weighted',
        zero_division=0
    )
    
    # Get top-k accuracies (k=1,3,5)
    top_k_accuracies = []
    for k in [1, 3, 5]:
        top_k = torch.topk(predictions, k, dim=1)[1]
        correct_k = top_k.eq(targets.view(-1, 1).expand_as(top_k)).float().sum(dim=1)
        top_k_acc = correct_k.mean().item()
        top_k_accuracies.append(top_k_acc)
    
    # Aggregate metrics
    metrics = {
        'accuracy': accuracy,
        'top1_acc': top_k_accuracies[0],
        'top3_acc': top_k_accuracies[1],
        'top5_acc': top_k_accuracies[2],
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def calculate_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: Predicted logits [batch_size, num_classes]
        targets: Target class indices [batch_size]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    pred_classes = torch.argmax(predictions, dim=1)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        device=predictions.device
    )
    
    # Fill confusion matrix
    for t, p in zip(targets, pred_classes):
        confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix

def calculate_per_class_metrics(
    confusion_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate per-class precision, recall, and F1 score.
    
    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        
    Returns:
        Tuple of (precision, recall, f1) tensors [num_classes]
    """
    # Calculate true positives, false positives, false negatives
    true_positives = torch.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives
    
    # Calculate metrics (adding epsilon to avoid division by zero)
    epsilon = 1e-7
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return precision, recall, f1
