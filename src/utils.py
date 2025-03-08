import os
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_image_path(image_path):
    """Validate image file path and format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        raise ValueError(f"Unsupported image format: {ext}")
    
    return True

def validate_parameters(params):
    """Validate input parameters."""
    required_params = ['n_segments', 'compactness', 'connectivity_threshold']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")
