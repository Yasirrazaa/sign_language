# Enhanced WLASL Framework

This repository contains an enhanced version of the WLASL (Word-Level American Sign Language) framework, incorporating memory-efficient preprocessing, comprehensive data analysis, cross-validation, and improved training capabilities while maintaining the original I3D and TGCN model architectures.

## New Features

- Memory-efficient video preprocessing
- Comprehensive data analysis tools
- Cross-validation support
- Enhanced training capabilities
- Better project structure
- Improved documentation

## Project Structure

```
WLASL/
├── configs/
│   └── base_config.py        # Centralized configuration
├── src/
│   ├── preprocessing/
│   │   └── video_processor.py    # Memory-efficient video processing
│   ├── data/
│   │   └── data_loader.py        # Data loading with cross-validation
│   └── training/
│       ├── trainer.py            # Enhanced training functionality
│       └── cross_validate.py     # Cross-validation implementation
├── code/
│   ├── I3D/                      # Original I3D implementation
│   └── TGCN/                     # Original TGCN implementation
├── analysis/                     # Data analysis outputs
├── logs/                         # Training logs
└── models/                       # Model checkpoints and weights
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/enhanced-wlasl.git
cd enhanced-wlasl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
# Preprocess videos and train I3D model with cross-validation
python main.py --model i3d --preprocess --cross-validate --data-path /path/to/data

# Train TGCN model without cross-validation
python main.py --model tgcn --data-path /path/to/data
```

## Memory-Efficient Processing

The enhanced framework includes several memory optimization features:

- Chunk-based video processing
- Efficient frame caching
- Mixed precision training
- Gradient accumulation
- Configurable batch sizes

Configure these in `configs/base_config.py`:

```python
PREPROCESSING_CONFIG = {
    'chunk_size': 32,          # Process videos in chunks
    'max_memory_usage': '80%', # Maximum memory usage
    'compression': 'JPEG',     # Frame compression format
    'compression_quality': 95  # JPEG quality
}
```

## Cross-Validation

The framework supports k-fold cross-validation with comprehensive metrics:

```python
from src.training.cross_validate import CrossValidator

validator = CrossValidator(
    model_class=model_class,
    model_params=model_params,
    data_info=data_info,
    num_folds=5
)

results = validator.run()
```

## Data Analysis

The framework automatically generates:

- Class distribution analysis
- Learning curves
- Confusion matrices
- Performance metrics
- Cross-validation results

Results are saved in the `analysis/` directory.

## Training Features

Enhanced training capabilities include:

- Learning rate scheduling
- Early stopping
- Mixed precision training
- Gradient clipping
- Weight decay
- Label smoothing

Configure training parameters in `configs/base_config.py`:

```python
TRAIN_CONFIG = {
    'num_epochs': 150,
    'learning_rate': 1e-4,
    'warmup_epochs': 5,
    'early_stopping_patience': 15,
    'mixed_precision': True
}
```

## Model Support

The framework maintains support for the original WLASL models while adding improvements:

### I3D Model
- Original implementation from WLASL
- Added memory optimizations
- Enhanced training features

### TGCN Model
- Original implementation from WLASL
- Improved data processing
- Added cross-validation support

## Converting from Original WLASL

If you're using the original WLASL dataset:

1. Place your videos in `data/raw_videos/`
2. Create a data info JSON file:
```json
[
    {
        "video_id": "video1",
        "label": 0,
        "frame_start": 1,
        "frame_end": 64
    }
]
```
3. Run preprocessing:
```bash
python main.py --preprocess --data-path /path/to/data
```

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

Licensed under the Computational Use of Data Agreement (C-UDA). Please refer to `C-UDA-1.0.pdf` for more information.

## Citation

If you use this enhanced framework, please cite both this repository and the original WLASL paper:

```bibtex
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
```

## Acknowledgments

This enhanced framework builds upon the original WLASL work by Li et al. We thank the original authors for their contributions to the sign language recognition field.
