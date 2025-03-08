import numpy as np
from PIL import Image
from skimage import color, exposure
import logging
import os

class ImagePreprocessor:
    def __init__(self):
        """Initialize preprocessor with parameters from the paper."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.logger = logging.getLogger(__name__)
        self.target_size = (750, 750)  # Standard size as per paper [52]
        self.clahe_clip_limit = 0.03  # CLAHE parameter from paper

    def load_image(self, image_path):
        """Load and validate image."""
        try:
            # Validate file extension
            _, ext = os.path.splitext(image_path)
            if ext.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {ext}")
            
            # Load image
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Step 1: Resample to 750x750 as specified in paper [52]
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            self.logger.info(f"Image resampled to {img.size} pixels")

            return np.array(img)

        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Error loading image: {str(e)}")

    def preprocess(self, image):
        """Preprocess image according to paper specifications."""
        try:
            # Ensure input is valid
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")

            # Handle different input formats
            if len(image.shape) == 2:  # Grayscale
                image = color.gray2rgb(image)
            elif image.shape[2] == 4:  # RGBA
                image = image[:,:,:3]
            elif len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")

            # Step 2: Convert to float and normalize to [0,1] range
            image = image.astype(float) / 255.0
            self.logger.info(f"Image normalized to range [0,1]: min={image.min():.3f}, max={image.max():.3f}")

            # Step 3: Apply CLAHE contrast enhancement with paper-specified parameters
            image = exposure.equalize_adapthist(image, clip_limit=self.clahe_clip_limit)
            self.logger.info(f"Applied CLAHE with clip_limit={self.clahe_clip_limit}")

            # Step 4: Convert to LAB color space as specified in paper
            image_lab = color.rgb2lab(image)

            # Step 5: Normalize LAB values to [0,1] range as per paper
            # L channel: [0, 100]
            # a channel: [-128, 127]
            # b channel: [-128, 127]
            l_chan = (image_lab[:,:,0]) / 100.0  # L channel normalization
            a_chan = (image_lab[:,:,1] + 128) / 255.0  # a channel normalization
            b_chan = (image_lab[:,:,2] + 128) / 255.0  # b channel normalization

            # Log channel statistics
            self.logger.info(f"LAB Normalization stats:")
            self.logger.info(f"L channel: min={l_chan.min():.3f}, max={l_chan.max():.3f}")
            self.logger.info(f"a channel: min={a_chan.min():.3f}, max={a_chan.max():.3f}")
            self.logger.info(f"b channel: min={b_chan.min():.3f}, max={b_chan.max():.3f}")

            # Stack normalized channels
            image_lab_normalized = np.dstack((l_chan, a_chan, b_chan))

            return image_lab_normalized

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise ValueError(f"Error during preprocessing: {str(e)}")
