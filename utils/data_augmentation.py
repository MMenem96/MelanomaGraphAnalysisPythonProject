import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

class DataAugmentation:
    """
    Safe data augmentation class for medical skin lesion images.
    Uses only horizontal flipping (2x augmentation).
    """
    
    def __init__(self, logger=None):
        """Initialize the DataAugmentation class."""
        self.logger = logger if logger else logging.getLogger(__name__)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def safe_bcc_augmentation(self, input_dir, output_dir, multiplier=2, apply_masks=False):
        """
        Safe data augmentation for BCC images using only horizontal flipping.
        
        Args:
            input_dir (str): Path to original BCC segmented images
            output_dir (str): Path to output augmented dataset
            multiplier (int): Target multiplication factor (default: 2x)
            apply_masks (bool): Not used - kept for compatibility
            
        Returns:
            tuple: (total_images_created, error_count)
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Creating augmented BCC dataset in: {output_dir}")
        
        # Get all image files
        image_files = self._get_image_files(input_dir)
        
        if len(image_files) == 0:
            self.logger.warning(f"No images found in {input_dir}")
            return 0, 0
        
        self.logger.info(f"Found {len(image_files)} images to augment")
        self.logger.info(f"Target: {len(image_files) * multiplier} total images ({multiplier}x augmentation)")
        
        augmented_count = 0
        error_count = 0
        
        
        for img_path in tqdm(image_files, desc="Augmenting BCC images"):
            try:
                # Read original image - NO PROCESSING
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # Keep everything unchanged
                if image is None:
                    self.logger.warning(f"Could not read image {img_path}")
                    error_count += 1
                    continue
                
                # Get filename components
                filename = img_path.stem
                extension = img_path.suffix
                
                # Generate ONLY flip - NO OTHER PROCESSING
                augmentations = self._generate_safe_augmentations(image, filename, extension)
                
                # Save with SAME QUALITY/FORMAT
                for aug_name, aug_image in augmentations.items():
                    output_path = os.path.join(output_dir, aug_name)
                    
                    # Save with maximum quality to preserve format
                    if extension.lower() in ['.jpg', '.jpeg']:
                        success = cv2.imwrite(output_path, aug_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    elif extension.lower() == '.png':
                        success = cv2.imwrite(output_path, aug_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        success = cv2.imwrite(output_path, aug_image)
                    
                    if success:
                        augmented_count += 1
                    else:
                        self.logger.warning(f"Failed to save {output_path}")
                        error_count += 1
                
                # SKIP mask processing to avoid any interference
                # mask_count = self._process_mask_files(img_path, input_dir, output_dir, filename, extension)
                        
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                error_count += 1
        
        # Generate summary
        self._print_augmentation_summary(len(image_files), augmented_count, error_count, output_dir)
        
        return augmented_count, error_count
    
    def _get_image_files(self, input_dir):
        """Get all image files from input directory."""
        input_path = Path(input_dir)
        image_files = []
        
        for ext in self.supported_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        return image_files
    
    def _generate_safe_augmentations(self, image, filename, extension):
        """Generate augmentations preserving the current image state."""
        augmentations = {}
        
        # Don't apply any additional processing - use image as-is
        augmentations[f"{filename}_original{extension}"] = image.copy()
        augmentations[f"{filename}_h_flip{extension}"] = cv2.flip(image, 1)
        
        return augmentations
    
    def _process_mask_files(self, img_path, input_dir, output_dir, filename, extension):
        """
        Process corresponding mask files if they exist.
        Only augments masks, doesn't apply them to images.
        """
        input_path = Path(input_dir)
        mask_count = 0
        
        # Common mask file naming patterns
        mask_patterns = [
            f"{filename}_mask{extension}",
            f"{filename}.mask{extension}",
            f"{filename}_segmentation{extension}",
            f"{filename}.segmentation{extension}",
            f"{filename}_seg{extension}",
            f"mask_{filename}{extension}",
            f"segmentation_{filename}{extension}"
        ]
        
        for mask_pattern in mask_patterns:
            mask_path = input_path / mask_pattern
            
            if mask_path.exists():
                try:
                    # Read mask (typically grayscale)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    if mask is not None:
                        # Generate 2x augmentations for mask (original + horizontal flip)
                        mask_augmentations = self._generate_mask_augmentations(mask, filename, extension)
                        
                        # Save augmented masks
                        for aug_name, aug_mask in mask_augmentations.items():
                            mask_output_path = os.path.join(output_dir, aug_name)
                            cv2.imwrite(mask_output_path, aug_mask)
                            mask_count += 1
                        
                        break  # Found and processed mask, no need to check other patterns
                        
                except Exception as e:
                    self.logger.warning(f"Error processing mask {mask_path}: {str(e)}")
        
        return mask_count
    
    def _generate_mask_augmentations(self, mask, filename, extension):
        """
        Generate 2x augmentations for mask files.
        
        Args:
            mask: Original mask array
            filename: Base filename
            extension: Original file extension
            
        Returns:
            dict: Dictionary with 2 masks - original and horizontal flip
        """
        mask_augmentations = {}
        
        # 1. Original mask
        mask_augmentations[f"{filename}_original_mask{extension}"] = mask.copy()
        
        # 2. Horizontal flip only
        h_flipped = cv2.flip(mask, 1)
        mask_augmentations[f"{filename}_h_flip_mask{extension}"] = h_flipped
        
        return mask_augmentations
    
    def _print_augmentation_summary(self, original_count, augmented_count, error_count, output_dir):
        """Print summary of augmentation results."""
        final_count = augmented_count
        multiplier = final_count / original_count if original_count > 0 else 0
        
        print("\n" + "="*60)
        print("🔬 BCC DATA AUGMENTATION SUMMARY")
        print("="*60)
        print(f"📊 Original images processed: {original_count}")
        print(f"🚀 Total augmented images: {final_count}")
        print(f"📈 Augmentation multiplier: {multiplier:.1f}x")
        print(f"❌ Errors encountered: {error_count}")
        print(f"📁 Output directory: {output_dir}")
        print("="*60)
        
        # Success rate
        success_rate = ((original_count - error_count) / original_count * 100) if original_count > 0 else 0
        print(f"✅ Success rate: {success_rate:.1f}%")
        
        if error_count == 0:
            print("🎯 Augmentation completed successfully!")
        else:
            print(f"⚠️  Augmentation completed with {error_count} errors.")
        
        print("="*60)
    
    def check_augmentation_results(self, output_dir):
        """
        Check and analyze the results of 2x augmentation.
        
        Args:
            output_dir (str): Directory containing augmented images
            
        Returns:
            dict: Statistics about augmentation results
        """
        
        if not os.path.exists(output_dir):
            self.logger.error(f"Directory {output_dir} does not exist!")
            return {}
        
        files = os.listdir(output_dir)
        
        # Count different types of augmented images (2x only)
        original_count = len([f for f in files if '_original' in f and 'mask' not in f])
        h_flip_count = len([f for f in files if '_h_flip' in f and 'mask' not in f])
        
        # Count mask files
        mask_count = len([f for f in files if 'mask' in f])
        
        # Total image count (should be 2x original)
        total_images = original_count + h_flip_count
        
        stats = {
            'original_images': original_count,
            'horizontal_flips': h_flip_count,
            'total_images': total_images,
            'mask_files': mask_count,
            'augmentation_factor': total_images / original_count if original_count > 0 else 0
        }
        
        # Print results
        print("\n" + "="*50)
        print("📊 AUGMENTATION RESULTS ANALYSIS")
        print("="*50)
        print(f"📷 Original images: {original_count}")
        print(f"↔️  Horizontal flips: {h_flip_count}")
        print(f"📁 Total images: {total_images}")
        print(f"🎭 Mask files: {mask_count}")
        print(f"📊 Augmentation factor: {stats['augmentation_factor']:.1f}x")
        print("="*50)
        
        return stats
    
    def validate_augmented_dataset(self, original_dir, augmented_dir):
        """
        Validate that 2x augmentation was successful and complete.
        
        Args:
            original_dir (str): Original dataset directory
            augmented_dir (str): Augmented dataset directory
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        
        try:
            # Check if directories exist
            if not os.path.exists(original_dir):
                self.logger.error(f"Original directory {original_dir} does not exist!")
                return False
                
            if not os.path.exists(augmented_dir):
                self.logger.error(f"Augmented directory {augmented_dir} does not exist!")
                return False
            
            # Count original images
            original_files = self._get_image_files(original_dir)
            original_count = len(original_files)
            
            # Count augmented images
            augmented_files = self._get_image_files(augmented_dir)
            augmented_count = len(augmented_files)
            
            # Expected count should be 2x original (original + horizontal flip)
            expected_count = original_count * 2
            
            print(f"\n🔍 DATASET VALIDATION:")
            print(f"Original images: {original_count}")
            print(f"Augmented images: {augmented_count}")
            print(f"Expected images: {expected_count}")
            
            if augmented_count >= expected_count:
                print("✅ Validation PASSED - 2x Augmentation is complete!")
                return True
            else:
                print(f"❌ Validation FAILED - Missing {expected_count - augmented_count} images")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False
    
    def cleanup_failed_augmentation(self, output_dir):
        """
        Clean up failed or incomplete augmentation attempts.
        
        Args:
            output_dir (str): Directory to clean up
        """
        
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                self.logger.info(f"Cleaned up directory: {output_dir}")
                print(f"🧹 Cleaned up failed augmentation directory: {output_dir}")
            except Exception as e:
                self.logger.error(f"Error cleaning up {output_dir}: {str(e)}")
        else:
            self.logger.info(f"Directory {output_dir} does not exist - nothing to clean up")

# Convenience function for direct usage
def augment_bcc_dataset(input_dir="data/bcc_segmented", 
                       output_dir="data/bcc_segmented_augmented",
                       multiplier=2):
    """
    Convenience function to augment BCC dataset with 2x horizontal flip.
    
    Args:
        input_dir (str): Input directory with original BCC images
        output_dir (str): Output directory for augmented images
        multiplier (int): Target augmentation multiplier (2x)
        
    Returns:
        tuple: (success, total_images, errors)
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create augmenter
    augmenter = DataAugmentation()
    
    print("🔬 Starting SAFE BCC Data Augmentation (2x Horizontal Flip)")
    print("Preserving full image size and original extensions")
    print("-" * 60)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' does not exist!")
        return False, 0, 1
    
    # Run augmentation
    total_images, errors = augmenter.safe_bcc_augmentation(input_dir, output_dir, multiplier)
    
    # Check results
    augmenter.check_augmentation_results(output_dir)
    
    # Validate
    success = augmenter.validate_augmented_dataset(input_dir, output_dir)
    
    if success and errors == 0:
        print("\n✅ 2x Augmentation completed successfully!")
        print(f"🎯 Ready to train with balanced dataset:")
        print(f"   Use --bcc-dir {output_dir}")
    else:
        print(f"\n⚠️  Augmentation completed with issues.")
        
    return success, total_images, errors

def main():
    """
    Main function to run 2x data augmentation directly.
    Configure your paths and settings here.
    """
    
    print("🔬 BCC DATA AUGMENTATION - 2x HORIZONTAL FLIP")
    print("="*50)
    
    # CONFIGURE YOUR PATHS HERE
    input_directory = "data/bcc_segmented"           # Your original BCC images
    output_directory = "data/bcc_segmented_augmented"  # Where to save augmented images
    multiplier = 2                                   # 2x augmentation (original + horizontal flip)
    cleanup_first = True                             # Clean output directory first
    
    # Display configuration
    print(f"📁 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")
    print(f"📊 Augmentation factor: {multiplier}x (horizontal flip only)")
    print(f"🧹 Cleanup first: {cleanup_first}")
    print(f"🖼️  Preserves: Full image size + original extensions")
    print("-" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create augmenter
        augmenter = DataAugmentation()
        
        # Cleanup if requested
        if cleanup_first and os.path.exists(output_directory):
            print(f"🧹 Cleaning up existing directory: {output_directory}")
            augmenter.cleanup_failed_augmentation(output_directory)
        
        # Check if input directory exists
        if not os.path.exists(input_directory):
            print(f"❌ ERROR: Input directory '{input_directory}' does not exist!")
            print(f"Please check the path and create the directory with your BCC images.")
            return False
        
        # Count original images
        original_files = augmenter._get_image_files(input_directory)
        print(f"📊 Found {len(original_files)} original BCC images")
        
        if len(original_files) == 0:
            print(f"❌ ERROR: No images found in '{input_directory}'")
            print(f"Please add your BCC images to this directory.")
            return False
        
        # Run 2x augmentation
        print(f"\n🚀 Starting 2x augmentation (horizontal flip only)...")
        total_images, errors = augmenter.safe_bcc_augmentation(
            input_dir=input_directory,
            output_dir=output_directory,
            multiplier=multiplier
        )
        
        # Check results
        print(f"\n📊 Analyzing results...")
        stats = augmenter.check_augmentation_results(output_directory)
        
        # Validate
        print(f"\n🔍 Validating 2x augmentation...")
        success = augmenter.validate_augmented_dataset(input_directory, output_directory)
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"🎯 FINAL RESULTS")
        print(f"="*60)
        
        if success and errors == 0:
            print(f"✅ SUCCESS! 2x Augmentation completed perfectly!")
            print(f"📈 Original images: {len(original_files)}")
            print(f"📈 Augmented images: {total_images}")
            print(f"📈 Multiplication factor: {total_images/len(original_files):.1f}x")
            print(f"\n🎯 Ready to train with balanced dataset!")
            print(f"   Use this command:")
            print(f"   python manual_train_features.py --bcc-dir {output_directory}")
        else:
            print(f"⚠️  COMPLETED WITH ISSUES:")
            print(f"   Total images: {total_images}")
            print(f"   Errors: {errors}")
            print(f"   Success: {success}")
        
        print(f"="*60)
        return success
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Main execution - this runs when you hit the run button
if __name__ == "__main__":
    
    # Direct run with main() function
    print("🔬 Starting BCC Data Augmentation (2x Horizontal Flip)...")
    success = main()
    
    if success:
        print("\n🎉 2x Augmentation completed successfully!")
        print("You can now train with the balanced dataset.")
    else:
        print("\n💥 Augmentation failed!")
        print("Check the error messages above.")
    
    # Keep the window open (optional)
    input("\nPress Enter to exit...")