import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import comb
import logging
import cv2

class MDFKTVisualizer:
    """Standalone MDFKT feature extractor and visualizer for paper figures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_image_and_mask(self, image_path):
        """Load image and create mask from segmented image."""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Ensure RGB (remove alpha if present)
            if img_array.ndim == 3 and img_array.shape[2] > 3:
                rgb_image = img_array[:, :, :3]
                # Use alpha channel as mask if available
                if img_array.shape[2] == 4:
                    mask = img_array[:, :, 3] > 0
                else:
                    # Create mask from non-black pixels
                    mask = np.any(rgb_image > 0, axis=2)
            elif img_array.ndim == 3:
                rgb_image = img_array[:, :, :3]
                mask = np.any(rgb_image > 0, axis=2)
            else:
                rgb_image = np.stack([img_array] * 3, axis=2)
                mask = img_array > 0
            
            # Normalize to [0, 1]
            if np.max(rgb_image) > 1.0:
                rgb_image = rgb_image.astype(float) / 255.0
                
            return rgb_image, mask
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
    
    def _krawtchouk_poly(self, n, x, N, p):
        """Compute raw Krawtchouk polynomial."""
        try:
            if p <= 0 or p >= 1:
                return 0.0
            
            if x < 0 or x > N or n < 0 or n > N:
                return 0.0
            
            q = 1.0 - p
            result = 0.0
            for j in range(n + 1):
                c1 = comb(x, j, exact=False)
                c2 = comb(N - x, n - j, exact=False)
                
                if j > 0:
                    power_term = (p / q) ** j
                    if power_term > 1e10:
                        power_term = 1e10
                    elif power_term < 1e-10:
                        power_term = 1e-10
                else:
                    power_term = 1.0
                
                term = ((-1) ** j) * c1 * c2 * power_term
                
                if np.isfinite(term):
                    result += term
            return result
        except Exception as e:
            self.logger.error(f"Error in Krawtchouk polynomial: {str(e)}")
            return 0.0
    
    def _normalized_krawtchouk(self, n, x, N, p):
        """Compute normalized Krawtchouk polynomial."""
        try:
            K_raw = self._krawtchouk_poly(n, x, N, p)
            
            # Normalization factor
            rho_n = ((-1) ** n) * np.sqrt(comb(N, n, exact=False)) * (p ** n) * ((1-p) ** (N - n))
            
            if abs(rho_n) < 1e-10:
                return 0.0
                
            K_normalized = K_raw / rho_n
            return K_normalized
        except Exception as e:
            self.logger.error(f"Error in normalized Krawtchouk: {str(e)}")
            return 0.0
    
    def _compute_K0_matrix_mdfkt(self, N, p):
        """Compute orthonormal Krawtchouk matrix K0 for MDFKT."""
        try:
            K = np.zeros((N, N), dtype=float)
            for n in range(N):
                for x in range(N):
                    K[n, x] = self._normalized_krawtchouk(n, x, N - 1, p)
            
            if not np.all(np.isfinite(K)):
                self.logger.warning("Non-finite values in K matrix")
                K = np.nan_to_num(K, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Orthonormalize via QR decomposition
            Q, R = np.linalg.qr(K.T)
            K0 = Q.T
            
            if not np.all(np.isfinite(K0)):
                self.logger.warning("QR produced non-finite values, using identity")
                return np.eye(N)
            
            return K0
        except Exception as e:
            self.logger.error(f"Error computing K0 matrix: {str(e)}")
            return np.eye(N)
    
    def _compute_lambda(self, N):
        """Compute lambda eigenvalues."""
        k = np.arange(1, N + 1)
        Lambda = 1.0 / k
        return Lambda
    
    def _apply_2D_MDFKT(self, K0, Lambda, f):
        """Apply 2D MDFKT forward transform."""
        try:
            if not np.all(np.isfinite(f)):
                f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = K0 @ f
            
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0, posinf=1e10, neginf=-1e10)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = Lambda[:, None] * temp
            
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0, posinf=1e10, neginf=-1e10)
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                Y = temp @ K0.T
            
            if not np.all(np.isfinite(Y)):
                Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return Y
        except Exception as e:
            self.logger.error(f"Error in 2D MDFKT: {str(e)}")
            return np.zeros_like(f, dtype=float)
    
    def _apply_2D_inverse_MDFKT(self, K0, Lambda, Y):
        """Apply 2D inverse MDFKT transform for reconstruction."""
        try:
            # Inverse: reverse the operations
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = Y @ K0
            
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Step 2: Remove Lambda scaling
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                temp = temp / Lambda[:, None]
            
            if not np.all(np.isfinite(temp)):
                temp = np.nan_to_num(temp, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Step 3: Row-wise inverse
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                f_reconstructed = K0.T @ temp
            
            if not np.all(np.isfinite(f_reconstructed)):
                f_reconstructed = np.nan_to_num(f_reconstructed, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return f_reconstructed
        except Exception as e:
            self.logger.error(f"Error in inverse MDFKT: {str(e)}")
            return np.zeros_like(Y, dtype=float)
    
    def extract_mdfkt_features(self, rgb_image, mask, N=64, p=0.5):
        """
        Extract MDFKT features from image with background removal.
        Matches conventional_features.py implementation.
        """
        try:
            # Get bounding box of the lesion (SAME AS conventional_features.py)
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                self.logger.warning("Invalid mask for MDFKT features")
                return None
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Extract lesion region from RGB image
            lesion_rgb = rgb_image[min_row:max_row+1, min_col:max_col+1].copy()
            
            # Zero out background (SAME AS conventional_features.py)
            lesion_mask = mask[min_row:max_row+1, min_col:max_col+1].copy()
            for channel_idx in range(3):
                lesion_rgb[:, :, channel_idx][~lesion_mask] = 0
            
            # Compute Krawtchouk matrix and Lambda
            K0 = self._compute_K0_matrix_mdfkt(N, p)
            Lambda = self._compute_lambda(N)
            
            results = {}
            
            # Process each RGB channel
            for i, channel_name in enumerate(['R', 'G', 'B']):
                # Extract channel from the cropped lesion (already zeroed)
                channel = lesion_rgb[:, :, i]
                
                # Resize channel to N x N for MDFKT
                channel_resized = cv2.resize(channel, (N, N), interpolation=cv2.INTER_LINEAR)
                
                # Check for valid input
                if not np.all(np.isfinite(channel_resized)):
                    self.logger.warning(f"Non-finite values in {channel_name} channel, cleaning")
                    channel_resized = np.nan_to_num(channel_resized, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Apply MDFKT
                Y = self._apply_2D_MDFKT(K0, Lambda, channel_resized)
                
                # Reconstruct for validation
                f_reconstructed = self._apply_2D_inverse_MDFKT(K0, Lambda, Y)
                
                # Resize reconstruction back to cropped lesion size
                f_reconstructed_resized = cv2.resize(
                    f_reconstructed, 
                    (lesion_rgb.shape[1], lesion_rgb.shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Store coefficients
                results[channel_name] = {
                    'coefficients': Y,
                    'magnitude': np.abs(Y),
                    'phase': np.angle(Y),
                    'real': np.real(Y),
                    'imaginary': np.imag(Y),
                    'reconstructed_N': f_reconstructed,  # N×N reconstruction
                    'reconstructed_cropped': f_reconstructed_resized  # Cropped size
                }
            
            # Store lesion info for later use
            results['lesion_bbox'] = (min_row, max_row, min_col, max_col)
            results['lesion_rgb'] = lesion_rgb
            results['lesion_mask'] = lesion_mask
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error extracting MDFKT features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_paper_figure(self, image_path, output_path='mdfkt_analysis.png', dpi=300, N=128):
        """
        Create publication-ready figure with 3×2 layout:
        Row 1: Original | Reconstructed | Magnitude
        Row 2: Phase | Real Part | Imaginary Part
        """
        try:
            # Load image and mask
            rgb_image, mask = self.load_image_and_mask(image_path)
            
            # Extract MDFKT features (with background removal)
            mdfkt_results = self.extract_mdfkt_features(rgb_image, mask, N=N, p=0.5)
            
            if mdfkt_results is None:
                raise ValueError("Failed to extract MDFKT features")
            
            # Get lesion bounding box and data
            min_row, max_row, min_col, max_col = mdfkt_results['lesion_bbox']
            lesion_rgb = mdfkt_results['lesion_rgb']
            
            # Reconstruct RGB image (cropped size with background zeroed)
            reconstructed_cropped = np.zeros_like(lesion_rgb)
            for i, ch in enumerate(['R', 'G', 'B']):
                reconstructed_cropped[:, :, i] = mdfkt_results[ch]['reconstructed_cropped']
            
            # Combine RGB channels for N×N visualization
            magnitude_combined = np.mean([mdfkt_results[ch]['magnitude'] for ch in ['R', 'G', 'B']], axis=0)
            phase_combined = np.mean([mdfkt_results[ch]['phase'] for ch in ['R', 'G', 'B']], axis=0)
            real_combined = np.mean([mdfkt_results[ch]['real'] for ch in ['R', 'G', 'B']], axis=0)
            imag_combined = np.mean([mdfkt_results[ch]['imaginary'] for ch in ['R', 'G', 'B']], axis=0)
            
            # Create figure with 2×3 layout
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1, Col 1: Original Image (cropped lesion)
            axes[0, 0].imshow(lesion_rgb)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Row 1, Col 2: Reconstructed Image (cropped lesion)
            axes[0, 1].imshow(np.clip(reconstructed_cropped, 0, 1))
            axes[0, 1].set_title('Reconstructed Image', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Row 1, Col 3: MDFKT Magnitude
            im1 = axes[0, 2].imshow(magnitude_combined, cmap='hot')
            axes[0, 2].set_title('MDFKT Magnitude', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Row 2, Col 1: MDFKT Phase
            im2 = axes[1, 0].imshow(phase_combined, cmap='twilight')
            axes[1, 0].set_title('MDFKT Phase', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Row 2, Col 2: MDFKT Real Part
            im3 = axes[1, 1].imshow(real_combined, cmap='RdYlGn')
            axes[1, 1].set_title('MDFKT Real Part', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Row 2, Col 3: MDFKT Imaginary Part
            im4 = axes[1, 2].imshow(imag_combined, cmap='PiYG')
            axes[1, 2].set_title('MDFKT Imaginary Part', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {output_path}")
            plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating paper figure: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    visualizer = MDFKTVisualizer()
    
    # Example: Process a BCC image
    image_path = "data/bcc_segmented_augmented/ISIC_0034095_segmented_original.png"  # Change to your image
    output_path = "mdfkt_paper_figure.png"
    
    # Create figure
    visualizer.create_paper_figure(image_path, output_path, dpi=300, N=128)