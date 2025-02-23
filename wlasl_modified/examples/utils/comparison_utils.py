"""Utilities for model comparison experiments."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
import time
from typing import Dict, List, Tuple, Any
import psutil
import gc
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

class ModelProfiler:
    """Profile model performance and memory usage."""
    
    def __init__(self, device: torch.device):
        """Initialize profiler."""
        self.device = device
        self.memory_stats = []
        self.timing_stats = []
    
    def profile_memory(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """Profile model memory usage."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        gc.collect()
        initial_memory = self._get_memory_stats()
        
        # Forward pass
        with torch.no_grad():
            _ = model(sample_input)
        
        peak_memory = self._get_memory_stats()
        memory_increase = {
            k: peak_memory[k] - initial_memory[k] 
            for k in peak_memory
        }
        
        return memory_increase
    
    def profile_speed(self,
                     model: torch.nn.Module,
                     sample_input: torch.Tensor,
                     num_iterations: int = 100,
                     warmup: int = 10) -> Dict[str, float]:
        """Profile model speed."""
        model.eval()
        timings = []
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Measure speed
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = model(sample_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            timings.append(time.time() - start_time)
        
        return {
            'mean_time': np.mean(timings),
            'std_time': np.std(timings),
            'min_time': np.min(timings),
            'max_time': np.max(timings)
        }
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            'ram_used': psutil.Process().memory_info().rss / 1024**3  # GB
        }
        
        if self.device.type == 'cuda':
            stats.update({
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3
            })
        
        return stats

class MetricsTracker:
    """Track and compute model metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset metric states."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self,
              predictions: torch.Tensor,
              targets: torch.Tensor,
              loss: Optional[float] = None):
        """Update metrics with batch results."""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='weighted'
        )
        
        metrics = {
            'accuracy': (predictions == targets).mean(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics

def load_config(config_path: Path) -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_model_configs(config: Dict) -> Dict[str, Dict]:
    """Prepare model-specific configurations."""
    model_configs = {}
    
    for model_name, model_config in config['models'].items():
        # Common settings
        model_configs[model_name] = {
            'num_classes': len(config['data'].get('classes', [])),
            'num_frames': config['data']['num_frames'],
            **model_config
        }
    
    return model_configs

class ExperimentLogger:
    """Log experiment results and visualizations."""
    
    def __init__(self, log_dir: Path, config: Dict):
        """Initialize logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Initialize wandb if enabled
        if config['logging']['wandb'].get('enabled', False):
            import wandb
            wandb.init(
                project=config['logging']['wandb']['project'],
                config=config,
                tags=config['logging']['wandb']['tags']
            )
    
    def log_metrics(self,
                   metrics: Dict[str, float],
                   step: Optional[int] = None,
                   prefix: str = ''):
        """Log metrics."""
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb if enabled
        if hasattr(self, 'wandb'):
            import wandb
            wandb.log(metrics, step=step)
        
        # Save to CSV
        df = pd.DataFrame([metrics])
        if step is not None:
            df['step'] = step
        
        csv_path = self.log_dir / 'metrics.csv'
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    
    def log_memory_stats(self, stats: Dict[str, float], model_name: str):
        """Log memory statistics."""
        stats = {f"memory/{k}": v for k, v in stats.items()}
        stats['model'] = model_name
        
        # Save to CSV
        df = pd.DataFrame([stats])
        csv_path = self.log_dir / 'memory_stats.csv'
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    
    def save_model_comparison(self, results: Dict[str, Dict]):
        """Save model comparison results."""
        # Convert to DataFrame
        df = pd.DataFrame(results).transpose()
        
        # Save detailed results
        df.to_csv(self.log_dir / 'model_comparison.csv')
        
        # Save summary plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            
            # Accuracy vs Memory plot
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=df, x='memory_usage', y='accuracy')
            for idx, row in df.iterrows():
                plt.annotate(idx, (row['memory_usage'], row['accuracy']))
            plt.title('Accuracy vs Memory Usage')
            
            # Speed vs Memory plot
            plt.subplot(1, 2, 2)
            sns.scatterplot(data=df, x='memory_usage', y='inference_time')
            for idx, row in df.iterrows():
                plt.annotate(idx, (row['memory_usage'], row['inference_time']))
            plt.title('Speed vs Memory Usage')
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'model_comparison.png')
            plt.close()
            
        except ImportError:
            logging.warning("Plotting requires matplotlib and seaborn")
    
    def close(self):
        """Close logger and cleanup."""
        if hasattr(self, 'wandb'):
            import wandb
            wandb.finish()

def setup_experiment(config_path: Path) -> Tuple[Dict, ExperimentLogger]:
    """Setup experiment with config and logger."""
    config = load_config(config_path)
    logger = ExperimentLogger(
        log_dir=Path(config['logging']['log_dir']),
        config=config
    )
    return config, logger