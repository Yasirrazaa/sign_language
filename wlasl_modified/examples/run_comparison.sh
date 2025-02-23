#!/bin/bash

# Script to run model comparison experiments

# Set environment variables
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print with color
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Check Python environment
if ! command -v python3 &> /dev/null; then
    print_color $RED "Error: Python 3 is required but not installed."
    exit 1
fi

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_color $GREEN "CUDA is available"
    GPU_FLAG=""
else
    print_color $YELLOW "CUDA not available, using CPU"
    GPU_FLAG="--cpu"
fi

# Create output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/comparison_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/visualizations

# Install requirements if needed
if [ ! -f "requirements_installed" ]; then
    print_color $YELLOW "Installing requirements..."
    pip install -r requirements_hybrid.txt
    touch requirements_installed
fi

# Run preprocessing if needed
PROCESSED_DIR="processed"
if [ ! -d "$PROCESSED_DIR" ]; then
    print_color $YELLOW "Running preprocessing..."
    python3 -m src.data.preprocessing \
        --input_dir data/videos \
        --output_dir $PROCESSED_DIR \
        --config examples/configs/comparison_config.yml
fi

# Run model comparison
print_color $GREEN "Starting model comparison..."

# Main experiment
python3 examples/run_comparison.py \
    --config examples/configs/comparison_config.yml \
    --output-dir $OUTPUT_DIR \
    $GPU_FLAG 2>&1 | tee $OUTPUT_DIR/logs/comparison.log

# Check if comparison was successful
if [ $? -eq 0 ]; then
    print_color $GREEN "\nModel comparison completed successfully!"
    print_color $GREEN "Results saved to: $OUTPUT_DIR"
    
    # Generate plots if matplotlib is available
    python3 -c "import matplotlib" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_color $GREEN "Generating visualization plots..."
        python3 examples/utils/plot_results.py \
            --results-dir $OUTPUT_DIR \
            --output-dir $OUTPUT_DIR/visualizations
    fi
    
    # Print summary
    echo -e "\nSummary of results:"
    cat $OUTPUT_DIR/logs/comparison.log | grep -A 4 "Model Comparison Summary:"
    
else
    print_color $RED "Error: Model comparison failed!"
    print_color $RED "Check logs at: $OUTPUT_DIR/logs/comparison.log"
    exit 1
fi

# Optional: Run notebooks for detailed analysis
if command -v jupyter &> /dev/null; then
    print_color $YELLOW "\nTo view detailed analysis, run:"
    echo "jupyter notebook examples/model_comparison.ipynb"
fi

# Print memory usage summary
print_color $YELLOW "\nMemory Usage Summary:"
if [ -f $OUTPUT_DIR/memory_stats.csv ]; then
    echo "Model Memory Usage (GB):"
    column -t -s, $OUTPUT_DIR/memory_stats.csv
fi

# Print final instructions
print_color $GREEN "\nTo view all results:"
echo "1. Check $OUTPUT_DIR/logs/comparison.log for detailed logs"
echo "2. View visualizations in $OUTPUT_DIR/visualizations/"
echo "3. Run the Jupyter notebook for interactive analysis"

# Cleanup temporary files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -r {} +