"""Run comprehensive model comparison experiment."""

import torch
import logging
from pathlib import Path
import argparse
from typing import Dict, List
import json
from datetime import datetime

from utils.comparison_utils import (
    ModelProfiler,
    MetricsTracker,
    setup_experiment
)

from src.models.hybrid_transformers import (
    CNNTransformer,
    TimeSformer,
    create_model as create_hybrid_model
)
from WLASL.code.I3D.models.pytorch_i3d import InceptionI3d
from src.models.efficient_sign_net import EfficientSignNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_models(config: Dict) -> Dict[str, torch.nn.Module]:
    """Create all model variants for comparison."""
    num_classes = config['data'].get('num_classes', 26)
    models = {}
    
    # Original I3D
    models['i3d'] = InceptionI3d(
        num_classes=num_classes,
        in_channels=3
    )
    
    # Memory-efficient version
    models['efficient_i3d'] = EfficientSignNet(
        num_classes=num_classes,
        in_channels=3,
        **config['models']['efficient_i3d']
    )
    
    # Hybrid models
    models['cnn_transformer'] = create_hybrid_model(
        'cnn_transformer',
        num_classes=num_classes,
        num_frames=config['data']['num_frames'],
        **config['models']['cnn_transformer']
    )
    
    models['timesformer'] = create_hybrid_model(
        'timesformer',
        num_classes=num_classes,
        num_frames=config['data']['num_frames'],
        **config['models']['timesformer']
    )
    
    return models

def evaluate_model_performance(
    model: torch.nn.Module,
    profiler: ModelProfiler,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate model performance metrics."""
    metrics = MetricsTracker()
    
    # Sample input for profiling
    sample_input = next(iter(data_loader))[0].to(device)
    
    # Profile memory and speed
    memory_stats = profiler.profile_memory(model, sample_input)
    speed_stats = profiler.profile_speed(
        model,
        sample_input,
        num_iterations=100,
        warmup=10
    )
    
    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        for frames, labels in data_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            predictions = torch.argmax(outputs, dim=1)
            
            metrics.update(predictions, labels)
    
    performance_metrics = metrics.compute()
    
    return {
        **performance_metrics,
        'memory_usage': memory_stats,
        'inference_speed': speed_stats
    }

def run_comparison(config_path: Path):
    """Run comprehensive model comparison."""
    # Setup experiment
    config, experiment_logger = setup_experiment(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    from src.data.loader import create_data_loaders
    data_loaders = create_data_loaders(
        data_dir=config['data']['video_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Initialize profiler
    profiler = ModelProfiler(device)
    
    # Create and evaluate models
    models = create_models(config)
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        model = model.to(device)
        
        try:
            # Evaluate model
            metrics = evaluate_model_performance(
                model,
                profiler,
                data_loaders['val'],
                device
            )
            
            results[name] = metrics
            
            # Log results
            experiment_logger.log_metrics(metrics, prefix=name)
            experiment_logger.log_memory_stats(metrics['memory_usage'], name)
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
            continue
        
        finally:
            # Cleanup
            del model
            torch.cuda.empty_cache()
    
    # Save comparison results
    experiment_logger.save_model_comparison(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(config['logging']['log_dir']) / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Memory Usage: {metrics['memory_usage']['gpu_allocated']:.2f} GB")
        print(f"Inference Time: {metrics['inference_speed']['mean_time']*1000:.2f} ms")
    
    # Close logger
    experiment_logger.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run model comparison")
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to comparison config file')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        run_comparison(Path(args.config))
        
    except KeyboardInterrupt:
        logger.info("Comparison interrupted by user")
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise

if __name__ == '__main__':
    main()