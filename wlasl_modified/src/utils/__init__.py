"""Utility functions and shared code."""

from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()

def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / 'data'

def get_video_dir() -> Path:
    """Get the video directory."""
    return get_project_root() / 'video'

def get_processed_dir() -> Path:
    """Get the processed data directory."""
    return get_project_root() / 'video'

def get_checkpoint_dir() -> Path:
    """Get the model checkpoints directory."""
    return get_project_root() / 'checkpoints'

def get_log_dir() -> Path:
    """Get the logs directory."""
    return get_project_root() / 'logs'

__all__ = [
    'get_project_root',
    'get_data_dir',
    'get_video_dir',
    'get_processed_dir',
    'get_checkpoint_dir',
    'get_log_dir'
]
