import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import cv2


class MDFKTImageAnalyzer:
    """
    Multi-parameter Discrete Fractional Krawtchouk Transform (MDFKT) for melanoma image analysis.
    Processes RGB channels separately and extracts Real, Imaginary, Phase, and Magnitude features.
    """
    

    def __init__(self, N: int = 128, p: float = 0.5):
        """
        Initialize the MDFKT analyzer.
        
        Args:
            N: Size of the transform (images will be resized to NxN)
            p: Parameter for Krawtchouk polynomials (default: 0.5)
        """
        self.N = N
        self.p = p
        print(f"Computing K0 matrix ({N}x{N})...")
        self.K0 = self._compute_K0_matrix(N, p)
        
        # Verify orthonormality
        print(f"Verifying K0 orthonormality...")
        verification = self.verify_orthonormality(self.K0)
        print(f"  Orthonormal: {verification['is_orthonormal']}")
        print(f"  Max deviation from identity: {verification['max_deviation_from_identity']:.2e}")
        print(f"  Max row norm deviation: {verification['max_row_norm_deviation']:.2e}")
        print(f"  KKT diagonal mean: {verification['KKT_diagonal_mean']:.6f}")
        print(f"  KKT off-diagonal max: {verification['KKT_offdiagonal_max']:.2e}")
        
        if not verification['is_orthonormal']:
            print("  ⚠️  WARNING: K0 matrix is NOT orthonormal within tolerance!")
        else:
            print("  ✓ K0 matrix is orthonormal")
        
        # Lambda eigenvalues: lambda_k = 1/k for k in [1, ..., N]
        # For k=0, we use 1.0 to avoid division by zero
        self.Lambda = np.zeros(N, dtype=complex)
        self.Lambda[0] = 1.0  # Special case for k=0
        for k in range(1, N):
            self.Lambda[k] = 1.0 / k
        
        print(f"Lambda computed: shape={self.Lambda.shape}, type={self.Lambda.dtype}")
        print(f"Lambda sample values (first 10): {self.Lambda[:10]}")
        print(f"Lambda formula: lambda_k = 1/k for k=1..{N-1}, lambda_0 = 1.0")

        
    def _krawtchouk_poly(self, n: int, x: int, N: int, p: float) -> float:
        """Compute Krawtchouk polynomial."""
        s = 0
        for j in range(0, n + 1):
            s += ((-1)**j *
                  comb(x, j) *
                  comb(N - x, n - j) *
                  (p / (1 - p))**j)
        return s
    
    def _normalized_krawtchouk(self, n: int, x: int, N: int, p: float) -> float:
        """Compute normalized Krawtchouk polynomial."""
        w_x = comb(N, x) * (p**x) * ((1 - p)**(N - x))
        norm = np.sqrt(comb(N, n) * (p**n) * ((1 - p)**(N - n)))
        return self._krawtchouk_poly(n, x, N, p) * np.sqrt(w_x) / norm
    
    def _compute_K0_matrix(self, N: int, p: float) -> np.ndarray:
            """Compute orthonormal Krawtchouk matrix K0."""
            K = np.zeros((N, N), dtype=float)
            for n in range(N):
                for x in range(N):
                    K[n, x] = self._normalized_krawtchouk(n, x, N - 1, p)
            # Orthonormalize via QR decomposition
            Q, R = np.linalg.qr(K.T)
            K0 = Q.T
            return K0
    
    def verify_orthonormality(self, K: np.ndarray, tolerance: float = 1e-10) -> Dict[str, any]:
        """
        Verify that K0 matrix is orthonormal.
        
        Args:
            K: Matrix to verify (should be K0)
            tolerance: Numerical tolerance for checks
            
        Returns:
            Dictionary with verification results
        """
        # K @ K.T should be identity
        KKT = K @ K.T
        identity = np.eye(K.shape[0])
        
        # Check if KKT is close to identity
        max_deviation = np.max(np.abs(KKT - identity))
        is_orthonormal = max_deviation < tolerance
        
        # Check row norms (should all be 1)
        row_norms = np.linalg.norm(K, axis=1)
        norm_deviation = np.max(np.abs(row_norms - 1.0))
        
        results = {
            'is_orthonormal': is_orthonormal,
            'max_deviation_from_identity': max_deviation,
            'max_row_norm_deviation': norm_deviation,
            'tolerance': tolerance,
            'all_row_norms_close_to_1': norm_deviation < tolerance,
            'KKT_diagonal_mean': np.mean(np.diag(KKT)),
            'KKT_offdiagonal_max': np.max(np.abs(KKT - np.diag(np.diag(KKT))))
        }
    
        return results
    

    def apply_2D_MDFKT(self, f: np.ndarray) -> np.ndarray:
        """
        Apply 2D MDFKT forward transform.
        
        Args:
            f: Input image (NxN array)
            
        Returns:
            Y: Transformed coefficients (complex NxN array)
        """
        temp = self.K0 @ f
        temp = self.Lambda[:, None] * temp
        Y = temp @ self.K0.T
        return Y
    
    def inverse_2D_MDFKT(self, Y: np.ndarray) -> np.ndarray:
        """
        Apply inverse 2D MDFKT.
        
        Args:
            Y: Transformed coefficients (complex NxN array)
            
        Returns:
            f_rec: Reconstructed image (NxN array)
        """
        inv_L = 1.0 / self.Lambda
        temp = Y @ self.K0
        temp = inv_L[:, None] * temp
        f_rec = self.K0.T @ temp
        return f_rec
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess an image (maintains RGB channels).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (original_resized_rgb, normalized_rgb)
        """
        # Load image in RGB
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Original image shape: {img_rgb.shape}")
        
        # Resize to NxN while maintaining 3 channels
        img_resized = cv2.resize(img_rgb, (self.N, self.N), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float64) / 255.0
        
        print(f"Resized to: {img_resized.shape}, Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        
        return img_resized, img_normalized
    
    def extract_channel_features(self, Y: np.ndarray, channel_name: str) -> Dict[str, float]:
        """
        Extract features from MDFKT coefficients for a single channel.
        
        Args:
            Y: MDFKT coefficients (complex)
            channel_name: Name of the channel (R, G, B)
            
        Returns:
            Dictionary of features with channel prefix
        """
        # Extract components as per your requirements
        real_part = Y.real
        imag_part = Y.imag
        magnitude = np.abs(Y)
        phase = np.angle(Y)
        
        features = {
            # Real part statistics
            f'{channel_name}_real_mean': np.mean(real_part),
            f'{channel_name}_real_std': np.std(real_part),
            f'{channel_name}_real_max': np.max(real_part),
            f'{channel_name}_real_min': np.min(real_part),
            
            # Imaginary part statistics
            f'{channel_name}_imag_mean': np.mean(imag_part),
            f'{channel_name}_imag_std': np.std(imag_part),
            f'{channel_name}_imag_max': np.max(imag_part),
            f'{channel_name}_imag_min': np.min(imag_part),
            
            # Magnitude (Abs) statistics
            f'{channel_name}_magnitude_mean': np.mean(magnitude),
            f'{channel_name}_magnitude_std': np.std(magnitude),
            f'{channel_name}_magnitude_max': np.max(magnitude),
            f'{channel_name}_magnitude_energy': np.sum(magnitude**2),
            
            # Phase statistics
            f'{channel_name}_phase_mean': np.mean(phase),
            f'{channel_name}_phase_std': np.std(phase),
            f'{channel_name}_phase_range': np.max(phase) - np.min(phase),
        }
        
        return features
    
    def analyze_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Complete analysis pipeline for a melanoma image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs (default: mdfkt_results in same folder)
            
        Returns:
            Dictionary containing analysis results for all channels
        """
        # Setup output directory
        if output_dir is None:
            output_dir = str(Path(image_path).parent / "mdfkt_results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print(f"MDFKT Analysis: {Path(image_path).name}")
        print("="*70)
        
        # Load and preprocess
        img_original, img_normalized = self.load_and_preprocess_image(image_path)
        
        # Process each RGB channel separately
        channels = ['R', 'G', 'B']
        results = {
            'original_image': img_original,
            'normalized_image': img_normalized,
            'mdfkt_coefficients': {},
            'reconstructed_images': {},
            'features': {},
            'reconstruction_errors': {}
        }
        
        all_features = {}
        
        for idx, channel_name in enumerate(channels):
            print(f"\n--- Processing {channel_name} Channel ---")
            
            # Extract channel
            f_channel = img_normalized[:, :, idx]
            print(f"Channel range: [{f_channel.min():.3f}, {f_channel.max():.3f}]")
            
            # Apply MDFKT
            print(f"Applying 2D MDFKT to {channel_name} channel...")
            Y = self.apply_2D_MDFKT(f_channel)
            results['mdfkt_coefficients'][channel_name] = Y
            
            # Reconstruct
            print(f"Reconstructing {channel_name} channel...")
            f_rec = self.inverse_2D_MDFKT(Y)
            results['reconstructed_images'][channel_name] = f_rec.real
            
            # Calculate reconstruction error
            reconstruction_error = np.max(np.abs(f_channel - f_rec.real))
            results['reconstruction_errors'][channel_name] = reconstruction_error
            print(f"Reconstruction error ({channel_name}): {reconstruction_error:.6e}")
            
            # Extract features
            channel_features = self.extract_channel_features(Y, channel_name)
            all_features.update(channel_features)
            results['features'][channel_name] = channel_features
            
            # Log features for this channel
            print(f"\n{channel_name} Channel Features:")
            for key, value in channel_features.items():
                print(f"  {key}: {value:.6f}")
        
        # Combine all features
        results['all_features'] = all_features
        
        # Display summary
        print("\n" + "="*70)
        print("SUMMARY - Reconstruction Errors:")
        for ch in channels:
            print(f"  {ch}: {results['reconstruction_errors'][ch]:.6e}")
        print("="*70)
        
        # Save visualizations
        self._save_visualizations(results, image_path, output_dir)
        
        # Save features to CSV
        features_df = pd.DataFrame([all_features])
        features_path = Path(output_dir) / f"{Path(image_path).stem}_features.csv"
        features_df.to_csv(features_path, index=False)
        print(f"\nFeatures saved to: {features_path}")
        
        return results
    
    def _save_visualizations(self, results: Dict, image_path: str, output_dir: str):
        """Save comprehensive visualization plots for all channels."""
        img_original = results['original_image']
        
        # Create main figure with all channels
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'MDFKT Analysis: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        channels = ['R', 'G', 'B']
        channel_colors = ['Reds', 'Greens', 'Blues']
        
        for idx, (channel_name, cmap_base) in enumerate(zip(channels, channel_colors)):
            Y = results['mdfkt_coefficients'][channel_name]
            f_rec = results['reconstructed_images'][channel_name]
            error = results['reconstruction_errors'][channel_name]
            
            row = idx * 2
            
            # Original channel
            plt.subplot(6, 4, row*4 + 1)
            plt.imshow(results['normalized_image'][:, :, idx], cmap=cmap_base, origin='lower', vmin=0, vmax=1)
            plt.title(f'{channel_name} - Original')
            plt.colorbar()
            plt.axis('off')
            
            # Reconstructed channel
            plt.subplot(6, 4, row*4 + 2)
            plt.imshow(f_rec, cmap=cmap_base, origin='lower', vmin=0, vmax=1)
            plt.title(f'{channel_name} - Reconstructed\nError: {error:.2e}')
            plt.colorbar()
            plt.axis('off')
            
            # Real part
            plt.subplot(6, 4, row*4 + 3)
            plt.imshow(Y.real, cmap='viridis', origin='lower')
            plt.title(f'{channel_name} - Real Part')
            plt.colorbar()
            plt.axis('off')
            
            # Imaginary part
            plt.subplot(6, 4, row*4 + 4)
            plt.imshow(Y.imag, cmap='plasma', origin='lower')
            plt.title(f'{channel_name} - Imaginary Part')
            plt.colorbar()
            plt.axis('off')
            
            # Magnitude
            plt.subplot(6, 4, row*4 + 5)
            plt.imshow(np.abs(Y), cmap='magma', origin='lower')
            plt.title(f'{channel_name} - Magnitude (Abs)')
            plt.colorbar()
            plt.axis('off')
            
            # Phase
            plt.subplot(6, 4, row*4 + 6)
            plt.imshow(np.angle(Y), cmap='twilight', origin='lower')
            plt.title(f'{channel_name} - Phase')
            plt.colorbar()
            plt.axis('off')
            
            # Magnitude spectrum (log scale)
            plt.subplot(6, 4, row*4 + 7)
            plt.imshow(np.log1p(np.abs(Y)), cmap='inferno', origin='lower')
            plt.title(f'{channel_name} - Log Magnitude')
            plt.colorbar()
            plt.axis('off')
            
            # Phase histogram
            plt.subplot(6, 4, row*4 + 8)
            plt.hist(np.angle(Y).flatten(), bins=50, color=channel_name.lower(), alpha=0.7)
            plt.title(f'{channel_name} - Phase Distribution')
            plt.xlabel('Phase (radians)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save main analysis plot
        output_path = Path(output_dir) / f"{Path(image_path).stem}_mdfkt_full_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Full analysis visualization saved to: {output_path}")
        plt.show()
        
        # Create a second figure for RGB composite
        fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle('RGB Composite Views', fontsize=14, fontweight='bold')
        
        # Original RGB
        axes[0, 0].imshow(img_original)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # Reconstructed RGB
        rec_rgb = np.stack([results['reconstructed_images'][ch] for ch in channels], axis=2)
        rec_rgb = np.clip(rec_rgb, 0, 1)
        axes[0, 1].imshow(rec_rgb)
        axes[0, 1].set_title('Reconstructed RGB')
        axes[0, 1].axis('off')
        
        # Combined magnitude
        mag_combined = np.stack([np.abs(results['mdfkt_coefficients'][ch]) for ch in channels], axis=2)
        mag_normalized = (mag_combined - mag_combined.min()) / (mag_combined.max() - mag_combined.min() + 1e-10)
        axes[0, 2].imshow(mag_normalized)
        axes[0, 2].set_title('Combined Magnitude (RGB)')
        axes[0, 2].axis('off')
        
        # Combined phase
        phase_combined = np.stack([np.angle(results['mdfkt_coefficients'][ch]) for ch in channels], axis=2)
        phase_normalized = (phase_combined - phase_combined.min()) / (phase_combined.max() - phase_combined.min() + 1e-10)
        axes[0, 3].imshow(phase_normalized)
        axes[0, 3].set_title('Combined Phase (RGB)')
        axes[0, 3].axis('off')
        
        # Individual channel magnitudes
        for idx, ch in enumerate(channels):
            axes[1, idx].imshow(np.abs(results['mdfkt_coefficients'][ch]), cmap='hot')
            axes[1, idx].set_title(f'{ch} Magnitude')
            axes[1, idx].axis('off')
        
        # Error visualization
        error_map = np.stack([np.abs(results['normalized_image'][:,:,i] - results['reconstructed_images'][ch]) 
                              for i, ch in enumerate(channels)], axis=2)
        axes[1, 3].imshow(error_map * 10)  # Amplified for visibility
        axes[1, 3].set_title('Reconstruction Error (×10)')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        output_path2 = Path(output_dir) / f"{Path(image_path).stem}_mdfkt_rgb_composite.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"RGB composite visualization saved to: {output_path2}")
        plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MDFKTImageAnalyzer(N=128, p=0.5)
    
    # Example: Analyze a melanoma image
    image_path = "data/bcc_segmented/ISIC_0033354_segmented.png"  
    
    try:
        results = analyzer.analyze_image(
            image_path=image_path,
            output_dir=None  # Will create 'mdfkt_results' folder automatically
        )
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Total features extracted: {len(results['all_features'])}")
        print(f"Check the output folder for visualizations and CSV file.")
        
    except FileNotFoundError:
        print(f"\n⚠️  Error: Image file not found!")
        print(f"Please update the 'image_path' variable with a valid melanoma image path.")
        print(f"Current path: {image_path}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()