"""
Skin Lesion Processing Module
This module provides the same functionality as the GUI but in a simple function-based interface.
It replicates all the processing steps from the Project_GUI.py file.

IMPORTANT: The segmented lesion area is extracted from the Gaussian blurred image 
(not the original image) to match the preprocessing used for mask generation.
This ensures consistency between the input used for segmentation and the final output.
"""

import os
import numpy as np
import pickle
import warnings
from pathlib import Path
from skimage.transform import resize
from skimage import io, img_as_ubyte, feature, color, measure
import cv2
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Activation, Dropout, Input, BatchNormalization

# Suppress TensorFlow warnings and specific skimage warnings
tf.config.run_functions_eagerly(True)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='skimage')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Suppress specific TensorFlow data warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set TensorFlow to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SkinLesionProcessor:
    def __init__(self, model_path='src/segmentation/model/model-skin-lesion-segmentation-org2000.h5'):
        """
        Initialize the processor with trained models
        
        Args:
            model_path (str): Path to the U-Net segmentation model
        """
        self.model_path = model_path
        self.model = None
        self.svm_model = None
        self.IMG_WIDTH = 384
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 3
        
        # Create output directory for intermediate results
        self.output_dir = Path("processing_outputs")
        self.output_dir.mkdir(exist_ok=False)
        
    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def get_unet(self, input_img, n_filters=16, dropout=0.1, batchnorm=True):
        """Function to define the UNET Model"""
        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    def load_models(self):
        """Load the trained models"""
        # Load U-Net model
        if self.model is None:
            if os.path.exists(self.model_path):
                # Suppress model loading warnings temporarily
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_img = Input((self.IMG_HEIGHT, self.IMG_WIDTH, 3), name='img')
                    self.model = self.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
                    self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
                    self.model.load_weights(self.model_path)
                print(f"U-Net model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"U-Net model file not found: {self.model_path}")
        

    def convert_to_grayscale(self, image):
        """Convert RGB image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def apply_blackhat_morphology(self, grayscale_image):
        """Apply blackhat morphological operation"""
        kernel = cv2.getStructuringElement(1, (17, 17))
        return cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, kernel)

    def apply_inpainting(self, original_image, blackhat_image):
        """Apply inpainting using blackhat mask"""
        ret, thresh2 = cv2.threshold(blackhat_image, 10, 255, cv2.THRESH_BINARY)
        return cv2.inpaint(original_image, thresh2, 1, cv2.INPAINT_TELEA)

    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (7, 7), 0)

    def get_segmentation_mask(self, preprocessed_image):
        """Get segmentation mask using U-Net model"""
        # Prepare input for the model
        X_test = np.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        
        # Resize and normalize the preprocessed image
        img = preprocessed_image[:, :, :self.IMG_CHANNELS]
        img_resized = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH, 3), mode='constant', preserve_range=True)
        X_test[0] = img_resized
        
        # Predict segmentation mask with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self.model.predict(X_test, verbose=0)
        predicted = (predicted > 0.5).astype(bool)
        
        return predicted

    def extract_features(self, input_dict, base_path):
        """
        Function to perform feature extraction for the segmented masks by calculating their 
        Asymmetry, Border irregularity, Color variation, Diameter and Texture
        """
        features = {}
        for idx, image_name in enumerate(input_dict):
            path = base_path / image_name
            image = io.imread(path)
            
            # Ensure image is in valid range and handle edge cases
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Use cv2 for grayscale conversion to avoid skimage warnings
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0  # Normalize to [0,1]
            else:
                gray_img = image / 255.0
                
            lesion_region = input_dict[image_name]

            # Asymmetry
            area_total = lesion_region.area
            img_mask = lesion_region.image
            horizontal_flip = np.fliplr(img_mask)
            diff_horizontal = img_mask * ~horizontal_flip
            vertical_flip = np.flipud(img_mask)
            diff_vertical = img_mask * ~vertical_flip
            diff_horizontal_area = np.count_nonzero(diff_horizontal)
            diff_vertical_area = np.count_nonzero(diff_vertical)
            asymm_idx = 0.5 * ((diff_horizontal_area / area_total) + (diff_vertical_area / area_total))
            ecc = lesion_region.eccentricity

            # Border irregularity
            compact_index = (lesion_region.perimeter ** 2) / (4 * np.pi * area_total)

            # Color variegation (with safety checks):
            sliced = image[lesion_region.slice]
            lesion_r = sliced[:, :, 0]
            lesion_g = sliced[:, :, 1]
            lesion_b = sliced[:, :, 2]
            
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-8
            max_r = np.max(lesion_r)
            max_g = np.max(lesion_g)
            max_b = np.max(lesion_b)
            
            C_r = np.std(lesion_r) / (max_r + epsilon) if max_r > 0 else 0
            C_g = np.std(lesion_g) / (max_g + epsilon) if max_g > 0 else 0
            C_b = np.std(lesion_b) / (max_b + epsilon) if max_b > 0 else 0

            # Diameter:
            eq_diameter = lesion_region.equivalent_diameter

            # Texture (with error handling):
            try:
                # Ensure gray_img is in proper format for texture analysis
                gray_img_ubyte = img_as_ubyte(np.clip(gray_img, 0, 1))
                glcm = feature.graycomatrix(image=gray_img_ubyte, distances=[1],
                                            angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 2],
                                            symmetric=True, normed=True)
                correlation = np.mean(feature.graycoprops(glcm, prop='correlation'))
                homogeneity = np.mean(feature.graycoprops(glcm, prop='homogeneity'))
                energy = np.mean(feature.graycoprops(glcm, prop='energy'))
                contrast = np.mean(feature.graycoprops(glcm, prop='contrast'))
            except Exception as e:
                print(f"Warning: Texture analysis failed for {image_name}: {e}")
                correlation = homogeneity = energy = contrast = 0.0

            features[image_name] = [asymm_idx, ecc, compact_index, C_r, C_g, C_b,
                                    eq_diameter, correlation, homogeneity, energy, contrast]
        return features


    def get_segmented_area(self, gaussian_blurred_image, predicted_mask):
        """Get the segmented lesion area from Gaussian blurred image"""
        # Resize mask to match gaussian blurred image dimensions
        mask = predicted_mask.squeeze().astype(np.uint8)
        mask_resized = cv2.resize(mask, (gaussian_blurred_image.shape[1], gaussian_blurred_image.shape[0]))
        
        # Create 3-channel mask
        mask_3d = np.stack([mask_resized, mask_resized, mask_resized], axis=2)
        
        # Apply mask to isolate the lesion from the Gaussian blurred image
        segmented_area = gaussian_blurred_image * mask_3d
        
        return segmented_area, mask_resized

    def get_isolated_lesion(self, original_image, predicted_mask):
        """Get the isolated lesion from original image (for comparison purposes)"""
        # Resize mask to match original image dimensions
        mask = predicted_mask.squeeze().astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # Create 3-channel mask
        mask_3d = np.stack([mask_resized, mask_resized, mask_resized], axis=2)
        
        # Apply mask to isolate the lesion from the original image
        isolated_lesion = original_image * mask_3d
        
        return isolated_lesion

    def process_image(self, image_path, save_intermediate=True):
        """
        Process an image through all steps and return all results
        
        Args:
            image_path (str): Path to the input image
            save_intermediate (bool): Whether to save intermediate processing steps
            
        Returns:
            dict: Dictionary containing all processing results:
                - 'original_image': Original input image
                - 'grayscale_image': Grayscale converted image
                - 'blackhat_image': Blackhat morphology result
                - 'inpainted_image': Inpainted image
                - 'gaussian_blurred_image': Gaussian blurred image
                - 'segmentation_mask': Binary segmentation mask
                - 'segmented_area': Gaussian blurred image with lesion isolated (main result)
                - 'isolated_lesion': Original image with lesion isolated (for comparison)
                - 'mask_binary': Binary mask resized to original image size
        """
        # Load models if not already loaded
        self.load_models()
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        
        # Step 1: Load original image
        original_image = io.imread(image_path)
        
        # Step 2: Convert to grayscale
        grayscale_image = self.convert_to_grayscale(original_image)
        
        # Step 3: Apply blackhat morphology
        blackhat_image = self.apply_blackhat_morphology(grayscale_image)
        
        # Step 4: Apply inpainting
        inpainted_image = self.apply_inpainting(original_image, blackhat_image)
        
        # Step 5: Apply Gaussian blur
        gaussian_blurred_image = self.apply_gaussian_blur(inpainted_image)
        
        # Step 6: Get segmentation mask
        predicted_mask = self.get_segmentation_mask(gaussian_blurred_image)
        
        # Step 7: Get segmented area (from Gaussian blurred image)
        segmented_area, mask_binary = self.get_segmented_area(gaussian_blurred_image, predicted_mask)
        
        # Step 8: Get isolated lesion (from original image for comparison)
        isolated_lesion = self.get_isolated_lesion(original_image, predicted_mask)
        
        # Save intermediate results if requested
        if save_intermediate:
            base_name = Path(image_path).stem
            io.imsave(self.output_dir / f"{base_name}_01_original.jpg", original_image)
            io.imsave(self.output_dir / f"{base_name}_02_grayscale.jpg", grayscale_image)
            io.imsave(self.output_dir / f"{base_name}_03_blackhat.jpg", blackhat_image)
            io.imsave(self.output_dir / f"{base_name}_04_inpainted.jpg", inpainted_image)
            io.imsave(self.output_dir / f"{base_name}_05_gaussian.jpg", gaussian_blurred_image)
            cv2.imwrite(str(self.output_dir / f"{base_name}_06_mask.jpg"), 
                       img_as_ubyte(predicted_mask.squeeze()))
            io.imsave(self.output_dir / f"{base_name}_07_segmented_area.jpg", 
                     segmented_area.astype(np.uint8))
            io.imsave(self.output_dir / f"{base_name}_08_isolated_lesion.jpg", 
                     isolated_lesion.astype(np.uint8))
        # Return all results
        return {
            'original_image': original_image,
            'grayscale_image': grayscale_image,
            'blackhat_image': blackhat_image,
            'inpainted_image': inpainted_image,
            'gaussian_blurred_image': gaussian_blurred_image,
            'segmentation_mask': predicted_mask,
            'segmented_area': segmented_area,
            'isolated_lesion': isolated_lesion,
            'mask_binary': mask_binary
        }


# Example usage
if __name__ == "__main__":
    # Initialize the processor once
    processor = SkinLesionProcessor()
    
    # Process an image
    try:
        image_path = "data/test/ISIC_0024306.jpg"  # Replace with your image path
        
        results = processor.process_image(image_path, save_intermediate=True)
        
        # Save final results
        io.imsave('final_segmented_lesion.jpg', results['segmented_area'].astype(np.uint8))
        io.imsave('final_isolated_lesion.jpg', results['isolated_lesion'].astype(np.uint8))
        cv2.imwrite('final_mask.jpg', results['mask_binary'] * 255)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model files and input image exist in the correct paths.")
