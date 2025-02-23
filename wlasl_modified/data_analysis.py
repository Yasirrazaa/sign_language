"""Module for analyzing sign language dataset."""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default matplotlib style
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.config import PROCESSED_DIR, DATA_CONFIG, VIZ_CONFIG, VIDEO_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Class to analyze sign language dataset."""
    
    def __init__(self, data: List[Dict]):
        """
        Initialize analyzer.
        
        Args:
            data: List of sign language data entries
        """
        self.data = data
        self.analysis_dir = PROCESSED_DIR / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_class_distribution(self) -> Dict:
        """
        Analyze distribution of classes in dataset.
        
        Returns:
            Dictionary containing class distribution statistics
        """
        # Count videos per class
        class_counts = defaultdict(int)
        class_splits = defaultdict(lambda: defaultdict(int))
        class_signers = defaultdict(set)
        
        for entry in self.data:
            gloss = entry['gloss']
            if entry.get('instances'):
                for instance in entry['instances']:
                    class_counts[gloss] += 1
                    class_splits[gloss][instance['split']] += 1
                    class_signers[gloss].add(instance['signer_id'])
        
        # Convert to sorted list of tuples
        class_data = [
            (gloss, count) for gloss, count in class_counts.items()
        ]
        class_data.sort(key=lambda x: x[1], reverse=True)
        
        # Plot distribution of top classes
        plt.figure(figsize=VIZ_CONFIG['plot_style']['figsize'])
        top_classes = class_data[:VIZ_CONFIG['max_classes_plot']]
        x = range(len(top_classes))
        plt.bar(x, [count for _, count in top_classes])
        plt.xticks(x, [gloss for gloss, _ in top_classes], rotation=45, ha='right')
        plt.title('Distribution of Top Classes')
        plt.xlabel('Class')
        plt.ylabel('Number of Videos')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'class_distribution.png')
        plt.close()
        
        # Save detailed statistics
        class_stats = {
            'total_classes': len(class_counts),
            'total_videos': sum(class_counts.values()),
            'mean_videos_per_class': np.mean(list(class_counts.values())),
            'median_videos_per_class': np.median(list(class_counts.values())),
            'min_videos_per_class': min(class_counts.values()),
            'max_videos_per_class': max(class_counts.values())
        }
        
        return class_stats

    def analyze_signer_distribution(self) -> Dict:
        """
        Analyze distribution of signers in dataset.
        
        Returns:
            Dictionary containing signer distribution statistics
        """
        signer_counts = defaultdict(int)
        signer_classes = defaultdict(set)
        
        for entry in self.data:
            if entry.get('instances'):
                for instance in entry['instances']:
                    signer_id = instance['signer_id']
                    signer_counts[signer_id] += 1
                    signer_classes[signer_id].add(entry['gloss'])
        
        # Plot signer statistics
        plt.figure(figsize=VIZ_CONFIG['plot_style']['figsize'])
        plt.hist(list(signer_counts.values()), bins=30)
        plt.title('Distribution of Videos per Signer')
        plt.xlabel('Number of Videos')
        plt.ylabel('Number of Signers')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'signer_distribution.png')
        plt.close()
        
        return {
            'total_signers': len(signer_counts),
            'mean_videos_per_signer': np.mean(list(signer_counts.values())),
            'mean_classes_per_signer': np.mean([len(c) for c in signer_classes.values()])
        }

    def analyze_split_distribution(self) -> Dict:
        """
        Analyze distribution of train/val/test splits.
        
        Returns:
            Dictionary containing split distribution statistics
        """
        split_counts = defaultdict(int)
        split_classes = defaultdict(set)
        
        for entry in self.data:
            if entry.get('instances'):
                for instance in entry['instances']:
                    split = instance['split']
                    split_counts[split] += 1
                    split_classes[split].add(entry['gloss'])
        
        # Plot split distribution
        plt.figure(figsize=VIZ_CONFIG['plot_style']['figsize'])
        splits = list(split_counts.keys())
        counts = list(split_counts.values())
        plt.bar(splits, counts)
        plt.title('Distribution of Dataset Splits')
        plt.xlabel('Split')
        plt.ylabel('Number of Videos')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'split_distribution.png')
        plt.close()
        
        return {
            'split_counts': dict(split_counts),
            'split_classes': {k: len(v) for k, v in split_classes.items()}
        }

    def select_classes(self) -> List[str]:
        """
        Select classes for training based on criteria.
        
        Returns:
            List of selected class names
        """
        # Count videos per class
        class_counts = defaultdict(int)
        class_splits = defaultdict(lambda: defaultdict(int))
        
        for entry in self.data:
            gloss = entry['gloss']
            if entry.get('instances'):
                for instance in entry['instances']:
                    class_counts[gloss] += 1
                    class_splits[gloss][instance['split']] += 1
        
        # Filter classes based on criteria
        selected_classes = []
        
        for gloss, count in class_counts.items():
            splits = class_splits[gloss]
            
            # Check criteria
            if (count >= DATA_CONFIG['min_videos_per_class'] and
                splits.get('train', 0) > 0 and
                splits.get('val', 0) > 0 and
                splits.get('test', 0) > 0):
                selected_classes.append(gloss)
        
        # Sort alphabetically
        selected_classes.sort()
        
        # Save selected classes
        with open(self.analysis_dir / 'selected_classes.json', 'w') as f:
            json.dump(selected_classes, f, indent=2)
        
        logger.info(f"Selected {len(selected_classes)} classes for training")
        return selected_classes

    def analyze_dataset(self) -> Dict:
        """
        Perform complete dataset analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Analyzing class distribution...")
        class_stats = self.analyze_class_distribution()
        
        logger.info("Analyzing signer distribution...")
        signer_stats = self.analyze_signer_distribution()
        
        logger.info("Analyzing split distribution...")
        split_stats = self.analyze_split_distribution()
        
        logger.info("Selecting classes...")
        selected_classes = self.select_classes()
        
        analysis_results = {
            'class_stats': class_stats,
            'signer_stats': signer_stats,
            'split_stats': split_stats,
            'num_selected_classes': len(selected_classes)
        }
        
        # Save analysis results
        with open(self.analysis_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results

def main():
    """Run dataset analysis."""
    try:
        # Load dataset
        with open(VIDEO_DIR / 'dataset.json', 'r') as f:
            data = json.load(f)
        
        # Create analyzer and analyze dataset
        analyzer = DataAnalyzer(data)
        results = analyzer.analyze_dataset()
        
        # Print summary
        print("\nDataset Analysis Summary:")
        print(f"Total Classes: {results['class_stats']['total_classes']}")
        print(f"Total Videos: {results['class_stats']['total_videos']}")
        print(f"Mean Videos per Class: {results['class_stats']['mean_videos_per_class']:.2f}")
        print(f"\nTotal Signers: {results['signer_stats']['total_signers']}")
        print(f"Mean Videos per Signer: {results['signer_stats']['mean_videos_per_signer']:.2f}")
        print(f"\nSplit Distribution:")
        for split, count in results['split_stats']['split_counts'].items():
            print(f"  {split}: {count}")
        print(f"\nSelected Classes: {results['num_selected_classes']}")
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise

if __name__ == '__main__':
    main()
