"""Generate visualizations from model comparison results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_results(results_dir: Path) -> Dict:
    """Load comparison results."""
    results_files = list(results_dir.glob('results_*.json'))
    if not results_files:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    # Load most recent results
    latest_results = max(results_files, key=lambda x: x.stat().st_mtime)
    with open(latest_results, 'r') as f:
        return json.load(f)

def plot_accuracy_comparison(results: Dict, output_dir: Path):
    """Plot accuracy comparison across models."""
    # Extract accuracies
    accuracies = {
        model: data['accuracy']
        for model, data in results.items()
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values())
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom'
        )
    
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png')
    plt.close()

def plot_memory_usage(results: Dict, output_dir: Path):
    """Plot memory usage comparison."""
    # Extract memory usage
    memory_data = []
    for model, data in results.items():
        memory_stats = data['memory_usage']
        memory_data.append({
            'model': model,
            'GPU Allocated': memory_stats['gpu_allocated'],
            'GPU Reserved': memory_stats['gpu_reserved'],
            'RAM': memory_stats.get('ram_used', 0)
        })
    
    df = pd.DataFrame(memory_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Memory usage bars
    x = np.arange(len(df))
    width = 0.25
    
    plt.bar(x - width, df['GPU Allocated'], width, label='GPU Allocated')
    plt.bar(x, df['GPU Reserved'], width, label='GPU Reserved')
    plt.bar(x + width, df['RAM'], width, label='RAM')
    
    plt.xlabel('Model')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage Comparison')
    plt.xticks(x, df['model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png')
    plt.close()

def plot_speed_comparison(results: Dict, output_dir: Path):
    """Plot inference speed comparison."""
    # Extract speed metrics
    speed_data = []
    for model, data in results.items():
        speed_stats = data['inference_speed']
        speed_data.append({
            'model': model,
            'mean_time': speed_stats['mean_time'] * 1000,  # Convert to ms
            'std_time': speed_stats['std_time'] * 1000
        })
    
    df = pd.DataFrame(speed_data)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['model'], df['mean_time'])
    
    # Add error bars
    plt.errorbar(
        df['model'],
        df['mean_time'],
        yerr=df['std_time'],
        fmt='none',
        color='black',
        capsize=5
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}ms',
            ha='center',
            va='bottom'
        )
    
    plt.title('Inference Speed Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_comparison.png')
    plt.close()

def plot_efficiency_frontier(results: Dict, output_dir: Path):
    """Plot accuracy vs memory efficiency frontier."""
    # Extract data
    data = []
    for model, metrics in results.items():
        data.append({
            'model': model,
            'accuracy': metrics['accuracy'],
            'memory': metrics['memory_usage']['gpu_allocated'],
            'speed': metrics['inference_speed']['mean_time'] * 1000
        })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 5))
    
    # Accuracy vs Memory
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='memory', y='accuracy', s=100)
    
    # Add labels
    for _, row in df.iterrows():
        plt.annotate(
            row['model'],
            (row['memory'], row['accuracy']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Accuracy vs Memory Usage')
    plt.xlabel('Memory Usage (GB)')
    plt.ylabel('Accuracy')
    
    # Accuracy vs Speed
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='speed', y='accuracy', s=100)
    
    # Add labels
    for _, row in df.iterrows():
        plt.annotate(
            row['model'],
            (row['speed'], row['accuracy']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Accuracy vs Inference Speed')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_frontier.png')
    plt.close()

def plot_all_metrics(results: Dict, output_dir: Path):
    """Generate all visualization plots."""
    try:
        logger.info("Plotting accuracy comparison...")
        plot_accuracy_comparison(results, output_dir)
        
        logger.info("Plotting memory usage comparison...")
        plot_memory_usage(results, output_dir)
        
        logger.info("Plotting speed comparison...")
        plot_speed_comparison(results, output_dir)
        
        logger.info("Plotting efficiency frontier...")
        plot_efficiency_frontier(results, output_dir)
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate result visualizations")
    
    parser.add_argument('--results-dir', type=str, required=True,
                      help='Directory containing results')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and plot results
    try:
        results = load_results(results_dir)
        plot_all_metrics(results, output_dir)
        logger.info(f"Plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main()