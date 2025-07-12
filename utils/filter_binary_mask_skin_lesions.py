#Read from Ham10000_binary_mask folder the images, and check if the image name of contains if the name of the image name in bcc or sk

import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

def filter_binary_masks():
    """
    Filter binary mask images based on whether their base names exist in bcc or sk folders.
    
    HAM10000_binary_mask images are named like: ISIC_0024306_segmentation.png
    BCC/SK images are named like: ISIC_0024312.jpg
    
    This function extracts the base name (e.g., ISIC_0024306) from mask images and
    copies them to bcc_filtered or sk_filtered folders based on where the corresponding
    original image exists.
    """
    
    # Define paths - get the project root directory
    script_dir = Path(__file__).parent  # utils directory
    project_root = script_dir.parent    # project root directory
    base_dir = project_root / "data"
    
    mask_dir = base_dir / "HAM10000_binary_mask"
    bcc_dir = base_dir / "bcc"
    sk_dir = base_dir / "sk"
    
    # Create output directories
    bcc_filtered_dir = base_dir / "bcc_filtered"
    sk_filtered_dir = base_dir / "sk_filtered"
    
    bcc_filtered_dir.mkdir(exist_ok=True)
    sk_filtered_dir.mkdir(exist_ok=True)
    
    # Get all image names from bcc and sk folders (without extensions)
    bcc_images = set()
    sk_images = set()
    
    if bcc_dir.exists():
        bcc_images = {os.path.splitext(f)[0] for f in os.listdir(bcc_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    
    if sk_dir.exists():
        sk_images = {os.path.splitext(f)[0] for f in os.listdir(sk_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    
    print(f"Found {len(bcc_images)} BCC images and {len(sk_images)} SK images")
    
    # Process binary mask images
    if not mask_dir.exists():
        print(f"Error: {mask_dir} directory not found!")
        return
    
    bcc_copied = 0
    sk_copied = 0
    not_found = 0
    
    for mask_file in os.listdir(mask_dir):
        if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Extract base name from mask file
        # ISIC_0024306_segmentation.png -> ISIC_0024306
        base_name = mask_file.replace('_segmentation', '').split('.')[0]
        
        mask_path = mask_dir / mask_file
        
        # Check if base name exists in bcc or sk folders
        if base_name in bcc_images:
            # Copy to bcc_filtered
            dest_path = bcc_filtered_dir / mask_file
            shutil.copy2(mask_path, dest_path)
            bcc_copied += 1
            print(f"Copied {mask_file} to bcc_filtered (matches {base_name})")
            
        elif base_name in sk_images:
            # Copy to sk_filtered
            dest_path = sk_filtered_dir / mask_file
            shutil.copy2(mask_path, dest_path)
            sk_copied += 1
            print(f"Copied {mask_file} to sk_filtered (matches {base_name})")
            
        else:
            not_found += 1
            print(f"No match found for {mask_file} (base: {base_name})")
    
    print(f"\nFiltering complete!")
    print(f"Copied {bcc_copied} masks to bcc_filtered")
    print(f"Copied {sk_copied} masks to sk_filtered")
    print(f"{not_found} masks had no matching images")

def filter_original_images():
    """
    Copy original images from bcc/sk folders that correspond to the filtered masks
    into bcc_filtered_originals and sk_filtered_originals folders.
    """
    
    # Define paths
    script_dir = Path(__file__).parent  # utils directory
    project_root = script_dir.parent    # project root directory
    base_dir = project_root / "data"
    
    bcc_dir = base_dir / "bcc"
    sk_dir = base_dir / "sk"
    bcc_filtered_dir = base_dir / "bcc_filtered"
    sk_filtered_dir = base_dir / "sk_filtered"
    
    # Create output directories for original images
    bcc_filtered_originals_dir = base_dir / "bcc_filtered_originals"
    sk_filtered_originals_dir = base_dir / "sk_filtered_originals"
    
    bcc_filtered_originals_dir.mkdir(exist_ok=True)
    sk_filtered_originals_dir.mkdir(exist_ok=True)
    
    print("Starting to filter original images...")
    
    # Process BCC filtered masks
    bcc_originals_copied = 0
    if bcc_filtered_dir.exists():
        for mask_file in os.listdir(bcc_filtered_dir):
            if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Extract base name from mask file
            # ISIC_0024306_segmentation.png -> ISIC_0024306
            base_name = mask_file.replace('_segmentation', '').split('.')[0]
            
            # Look for corresponding original image in bcc folder
            for original_file in os.listdir(bcc_dir):
                if original_file.startswith(base_name) and original_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Copy original image
                    src_path = bcc_dir / original_file
                    dest_path = bcc_filtered_originals_dir / original_file
                    shutil.copy2(src_path, dest_path)
                    bcc_originals_copied += 1
                    print(f"Copied BCC original: {original_file}")
                    break
    
    # Process SK filtered masks
    sk_originals_copied = 0
    if sk_filtered_dir.exists():
        for mask_file in os.listdir(sk_filtered_dir):
            if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Extract base name from mask file
            # ISIC_0024306_segmentation.png -> ISIC_0024306
            base_name = mask_file.replace('_segmentation', '').split('.')[0]
            
            # Look for corresponding original image in sk folder
            for original_file in os.listdir(sk_dir):
                if original_file.startswith(base_name) and original_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Copy original image
                    src_path = sk_dir / original_file
                    dest_path = sk_filtered_originals_dir / original_file
                    shutil.copy2(src_path, dest_path)
                    sk_originals_copied += 1
                    print(f"Copied SK original: {original_file}")
                    break
    
    print(f"\nOriginal image filtering complete!")
    print(f"Copied {bcc_originals_copied} BCC original images to bcc_filtered_originals")
    print(f"Copied {sk_originals_copied} SK original images to sk_filtered_originals")

def merge_original_with_mask():
    """
    Merge original images with their corresponding binary masks side by side.
    Creates visualization with labels and saves to merged folders.
    """
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_dir = project_root / "data"
    
    bcc_filtered_originals_dir = base_dir / "bcc_filtered_originals"
    sk_filtered_originals_dir = base_dir / "sk_filtered_originals"
    bcc_filtered_dir = base_dir / "bcc_filtered"
    sk_filtered_dir = base_dir / "sk_filtered"
    
    # Create output directories for merged images
    bcc_merged_dir = base_dir / "bcc_merged_original_binary_mask"
    sk_merged_dir = base_dir / "sk_merged_original_binary_mask"
    
    bcc_merged_dir.mkdir(exist_ok=True)
    sk_merged_dir.mkdir(exist_ok=True)
    
    print("Starting to merge original images with binary masks...")
    
    # Process BCC images
    bcc_merged_count = 0
    if bcc_filtered_originals_dir.exists() and bcc_filtered_dir.exists():
        for original_file in os.listdir(bcc_filtered_originals_dir):
            if not original_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Get base name (e.g., ISIC_0024306 from ISIC_0024306.jpg)
            base_name = os.path.splitext(original_file)[0]
            
            # Look for corresponding mask file
            mask_file = f"{base_name}_segmentation.png"
            mask_path = bcc_filtered_dir / mask_file
            original_path = bcc_filtered_originals_dir / original_file
            
            if mask_path.exists():
                # Create merged image
                merged_image_path = bcc_merged_dir / f"{base_name}_merged.png"
                create_merged_image(original_path, mask_path, merged_image_path, base_name, "BCC")
                bcc_merged_count += 1
                print(f"Created BCC merged image: {base_name}_merged.png")
    
    # Process SK images
    sk_merged_count = 0
    if sk_filtered_originals_dir.exists() and sk_filtered_dir.exists():
        for original_file in os.listdir(sk_filtered_originals_dir):
            if not original_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Get base name (e.g., ISIC_0024306 from ISIC_0024306.jpg)
            base_name = os.path.splitext(original_file)[0]
            
            # Look for corresponding mask file
            mask_file = f"{base_name}_segmentation.png"
            mask_path = sk_filtered_dir / mask_file
            original_path = sk_filtered_originals_dir / original_file
            
            if mask_path.exists():
                # Create merged image
                merged_image_path = sk_merged_dir / f"{base_name}_merged.png"
                create_merged_image(original_path, mask_path, merged_image_path, base_name, "SK")
                sk_merged_count += 1
                print(f"Created SK merged image: {base_name}_merged.png")
    
    print(f"\nMerging complete!")
    print(f"Created {bcc_merged_count} BCC merged images in bcc_merged_original_binary_mask")
    print(f"Created {sk_merged_count} SK merged images in sk_merged_original_binary_mask")

def create_merged_image(original_path, mask_path, output_path, image_name, image_type):
    """
    Create a side-by-side merged image with original and binary mask.
    
    Args:
        original_path: Path to original image
        mask_path: Path to binary mask
        output_path: Path to save merged image
        image_name: Name of the image for title
        image_type: Type (BCC or SK) for title
    """
    try:
        # Read original image
        original = cv2.imread(str(original_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Read mask image
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize images to same height if needed
        if original.shape[:2] != mask.shape:
            height = min(original.shape[0], mask.shape[0])
            width_orig = int(original.shape[1] * height / original.shape[0])
            width_mask = int(mask.shape[1] * height / mask.shape[0])
            
            original = cv2.resize(original, (width_orig, height))
            mask = cv2.resize(mask, (width_mask, height))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Display binary mask
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Binary Mask', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add main title
        fig.suptitle(f'{image_type} - {image_name}', fontsize=16, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating merged image for {image_name}: {str(e)}")

if __name__ == "__main__":
    # print("Step 1: Filtering binary masks...")
    # filter_binary_masks()
    
    # print("\nStep 2: Filtering original images...")
    # filter_original_images()
    
    print("\nStep 3: Merging original images with binary masks...")
    merge_original_with_mask()
