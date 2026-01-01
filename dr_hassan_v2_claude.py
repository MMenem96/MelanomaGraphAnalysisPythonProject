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
    

    def __init__(self, N: int = 64, p: float = 0.5, lambda_method: str = 'inverse_reciprocal'):
        """
        Initialize the MDFKT analyzer.
        
        Args:
            N: Size of the transform (images will be resized to NxN)
            p: Parameter for Krawtchouk polynomials (default: 0.5)
            lambda_method: Method for Lambda eigenvalues (default: 'mdfkt')
                          Options: 'mdfkt', 'reciprocal', 'inverse_reciprocal', 'dft', 'odd_harmonics'
        """
        self.N = N
        self.p = p
        self.lambda_method = lambda_method
        
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
        
        # Generate Lambda eigenvalues using the specified method
        print(f"\nGenerating Lambda eigenvalues...")
        self.Lambda = self.get_lambda_eigenvalues(N, lambda_method)
        print()

    def get_lambda_eigenvalues(self, N: int, method: str = 'reciprocal') -> np.ndarray:
        """
        Generate Lambda eigenvalues according to different formulas.
        
        Args:
            N: Size of the transform (number of eigenvalues)
            method: Method to use for generating eigenvalues
                    Options:
                    - 'mdfkt': exp(i·π·k/N) - Half circle (default, fractional)
                    - 'reciprocal': 1/k - reciprocal eigenvalues
                    - 'inverse_reciprocal': 1/(N-k) - Inverse reciprocal
                    - 'dft': exp(i·2π·k/N) - Full circle (standard DFT)
                    - 'odd_harmonics': exp(i·(2k+1)·π/N) - Odd harmonics (DCT-like)
        
        Returns:
            np.ndarray: Array of Lambda eigenvalues of shape (N,)
        
        Raises:
            ValueError: If invalid method is specified
        """
        k = np.arange(N)
        
        if method == 'mdfkt':
            # Default MDFKT: Half circle on unit circle
            # λₖ = exp(i·π·k/N) for k=0..N-1
            Lambda = np.exp(1j * np.pi * k / N)
            print(f"Lambda method: MDFKT (Half circle)")
            print(f"  Formula: λₖ = exp(i·π·k/N)")
            print(f"  Type: Complex (unit circle)")
            print(f"  First 5 values: {Lambda[:5]}")
            
        elif method == 'reciprocal':
            # reciprocal eigenvalues (low-frequency emphasis)
            # λₖ = 1/k for k=1..N
            Lambda = 1.0 / (k + 1)
            print(f"Lambda method: reciprocal")
            print(f"  Formula: λₖ = 1/k")
            print(f"  Type: Real (reciprocaling)")
            print(f"  First 5 values: {Lambda[:5]}")
            
        elif method == 'inverse_reciprocal':
            # Inverse reciprocal (high-frequency emphasis)
            # λₖ = 1/(N-k) for k=0..N-1
            Lambda = 1.0 / (N - k)
            print(f"Lambda method: Inverse Reciprocal")
            print(f"  Formula: λₖ = 1/(N-k)")
            print(f"  Type: Real (increasing)")
            print(f"  First 5 values: {Lambda[:5]}")
            
        elif method == 'dft':
            # Standard DFT: Full circle on unit circle
            # λₖ = exp(i·2π·k/N) for k=0..N-1
            Lambda = np.exp(1j * 2 * np.pi * k / N)
            print(f"Lambda method: DFT (Full circle)")
            print(f"  Formula: λₖ = exp(i·2π·k/N)")
            print(f"  Type: Complex (unit circle)")
            print(f"  First 5 values: {Lambda[:5]}")
            
        elif method == 'odd_harmonics':
            # Odd harmonics (DCT-like behavior)
            # λₖ = exp(i·(2k+1)·π/N) for k=0..N-1
            Lambda = np.exp(1j * (2 * k + 1) * np.pi / N)
            print(f"Lambda method: Odd Harmonics")
            print(f"  Formula: λₖ = exp(i·(2k+1)·π/N)")
            print(f"  Type: Complex (unit circle)")
            print(f"  First 5 values: {Lambda[:5]}")
            
        else:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'mdfkt', 'reciprocal', 'inverse_reciprocal', 'dft', 'odd_harmonics'")
        
        print(f"  Shape: {Lambda.shape}, Dtype: {Lambda.dtype}")
        print(f"  Range: [{np.min(np.abs(Lambda)):.6f}, {np.max(np.abs(Lambda)):.6f}]")
        
        return Lambda


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
        # Extract components
        real_part = Y.real
        magnitude = np.abs(Y)
        
        # Check if lambda is imaginary (has non-zero imaginary component)
        has_imaginary_lambda = np.any(self.Lambda.imag != 0)
        
        features = {
            # Real part statistics
            f'{channel_name}_real_mean': np.mean(real_part),
            f'{channel_name}_real_std': np.std(real_part),
            f'{channel_name}_real_max': np.max(real_part),
            f'{channel_name}_real_min': np.min(real_part),
            
            # Magnitude (Abs) statistics
            f'{channel_name}_magnitude_mean': np.mean(magnitude),
            f'{channel_name}_magnitude_std': np.std(magnitude),
            f'{channel_name}_magnitude_max': np.max(magnitude),
            f'{channel_name}_magnitude_energy': np.sum(magnitude**2),
        }
        
        # Only add imaginary and phase features if lambda is imaginary
        if has_imaginary_lambda:
            imag_part = Y.imag
            phase = np.angle(Y)
            
            features.update({
                # Imaginary part statistics
                f'{channel_name}_imag_mean': np.mean(imag_part),
                f'{channel_name}_imag_std': np.std(imag_part),
                f'{channel_name}_imag_max': np.max(imag_part),
                f'{channel_name}_imag_min': np.min(imag_part),
                
                # Phase statistics
                f'{channel_name}_phase_mean': np.mean(phase),
                f'{channel_name}_phase_std': np.std(phase),
                f'{channel_name}_phase_range': np.max(phase) - np.min(phase),
            })
        
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
            'features': {}
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
        print("FEATURE EXTRACTION COMPLETE")
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
        
        # Check if lambda is imaginary to determine which visualizations to show
        has_imaginary_lambda = np.any(self.Lambda.imag != 0)
        
        # ========== FIGURE 1: Combined view of all channels ==========
        if has_imaginary_lambda:
            fig = plt.figure(figsize=(16, 12))
            subplot_layout = (3, 4)  # 3 rows, 4 columns for R, G, B with imag and phase
        else:
            fig = plt.figure(figsize=(12, 9))
            subplot_layout = (3, 3)  # 3 rows, 3 columns for R, G, B without imag and phase
        
        fig.suptitle(f'MDFKT Analysis: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        channels = ['R', 'G', 'B']
        
        for idx, channel_name in enumerate(channels):
            Y = results['mdfkt_coefficients'][channel_name]
            
            if has_imaginary_lambda:
                # With imaginary lambda: show original, real, imaginary, magnitude
                base_idx = idx * 4
                
                # Original channel
                plt.subplot(3, 4, base_idx + 1)
                plt.imshow(results['normalized_image'][:, :, idx], cmap='viridis', origin='lower', vmin=0, vmax=1)
                plt.title(f'{channel_name} - Original')
                plt.colorbar()
                plt.axis('off')
                
                # Real part
                plt.subplot(3, 4, base_idx + 2)
                plt.imshow(Y.real, cmap='viridis', origin='lower')
                plt.title(f'{channel_name} - Real')
                plt.colorbar()
                plt.axis('off')
                
                # Imaginary part
                plt.subplot(3, 4, base_idx + 3)
                plt.imshow(Y.imag, cmap='plasma', origin='lower')
                plt.title(f'{channel_name} - Imaginary')
                plt.colorbar()
                plt.axis('off')
                
                # Magnitude
                plt.subplot(3, 4, base_idx + 4)
                plt.imshow(np.abs(Y), cmap='magma', origin='lower')
                plt.title(f'{channel_name} - Magnitude')
                plt.colorbar()
                plt.axis('off')
            else:
                # Without imaginary lambda: show original, real, magnitude only
                base_idx = idx * 3
                
                # Original channel
                plt.subplot(3, 3, base_idx + 1)
                plt.imshow(results['normalized_image'][:, :, idx], cmap='viridis', origin='lower', vmin=0, vmax=1)
                plt.title(f'{channel_name} - Original')
                plt.colorbar()
                plt.axis('off')
                
                # Real part
                plt.subplot(3, 3, base_idx + 2)
                plt.imshow(Y.real, cmap='viridis', origin='lower')
                plt.title(f'{channel_name} - Real')
                plt.colorbar()
                plt.axis('off')
                
                # Magnitude
                plt.subplot(3, 3, base_idx + 3)
                plt.imshow(np.abs(Y), cmap='magma', origin='lower')
                plt.title(f'{channel_name} - Magnitude')
                plt.colorbar()
                plt.axis('off')
            
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        output_path = Path(output_dir) / f"{Path(image_path).stem}_mdfkt_full_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Full analysis visualization saved to: {output_path}")
        plt.show()
        plt.close()
        
        # ========== FIGURE 2: RGB Composite ==========
        if has_imaginary_lambda:
            fig2, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig2.suptitle('RGB Composite Views', fontsize=14, fontweight='bold')
            
            # Original RGB
            axes[0, 0].imshow(img_original)
            axes[0, 0].set_title('Original RGB')
            axes[0, 0].axis('off')
            
            # Combined magnitude
            mag_r = np.abs(results['mdfkt_coefficients']['R'])
            mag_g = np.abs(results['mdfkt_coefficients']['G'])
            mag_b = np.abs(results['mdfkt_coefficients']['B'])
            
            mag_r_norm = (mag_r - mag_r.min()) / (mag_r.max() - mag_r.min() + 1e-10)
            mag_g_norm = (mag_g - mag_g.min()) / (mag_g.max() - mag_g.min() + 1e-10)
            mag_b_norm = (mag_b - mag_b.min()) / (mag_b.max() - mag_b.min() + 1e-10)
            
            mag_combined = np.stack([mag_r_norm, mag_g_norm, mag_b_norm], axis=2)
            axes[0, 1].imshow(mag_combined)
            axes[0, 1].set_title('Combined Magnitude (RGB)')
            axes[0, 1].axis('off')
            
            # Combined phase
            phase_combined = np.stack([np.angle(results['mdfkt_coefficients'][ch]) for ch in channels], axis=2)
            phase_normalized = (phase_combined + np.pi) / (2 * np.pi)
            axes[0, 2].imshow(phase_normalized)
            axes[0, 2].set_title('Combined Phase (RGB)')
            axes[0, 2].axis('off')
            
            # Individual channel magnitudes
            for idx, ch in enumerate(channels):
                mag = np.abs(results['mdfkt_coefficients'][ch])
                axes[1, idx].imshow(mag, cmap='hot')
                axes[1, idx].set_title(f'{ch} Magnitude')
                axes[1, idx].axis('off')
        else:
            fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig2.suptitle('RGB Composite Views', fontsize=14, fontweight='bold')
            
            # Original RGB
            axes[0].imshow(img_original)
            axes[0].set_title('Original RGB')
            axes[0].axis('off')
            
            # Combined magnitude
            mag_r = np.abs(results['mdfkt_coefficients']['R'])
            mag_g = np.abs(results['mdfkt_coefficients']['G'])
            mag_b = np.abs(results['mdfkt_coefficients']['B'])
            
            mag_r_norm = (mag_r - mag_r.min()) / (mag_r.max() - mag_r.min() + 1e-10)
            mag_g_norm = (mag_g - mag_g.min()) / (mag_g.max() - mag_g.min() + 1e-10)
            mag_b_norm = (mag_b - mag_b.min()) / (mag_b.max() - mag_b.min() + 1e-10)
            
            mag_combined = np.stack([mag_r_norm, mag_g_norm, mag_b_norm], axis=2)
            axes[1].imshow(mag_combined)
            axes[1].set_title('Combined Magnitude (RGB)')
            axes[1].axis('off')
        
        plt.tight_layout()
        output_path2 = Path(output_dir) / f"{Path(image_path).stem}_mdfkt_rgb_composite.png"
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"RGB composite visualization saved to: {output_path2}")
        plt.show()
        plt.close()
        
        # ========== FIGURES 3-5: Individual Channel Analysis (Publication Ready) ==========
        for idx, channel_name in enumerate(channels):
            Y = results['mdfkt_coefficients'][channel_name]
            f_original = results['normalized_image'][:, :, idx]
            
            
            # Create figure based on lambda type
            if has_imaginary_lambda:
                fig_ch, axes_ch = plt.subplots(2, 3, figsize=(15, 10))
                fig_ch.suptitle(f'MDFKT Analysis - {channel_name} Channel: {Path(image_path).name}', 
                               fontsize=16, fontweight='bold', y=0.98)
                
                # Row 1, Col 1: Original
                im1 = axes_ch[0, 0].imshow(f_original, cmap='viridis', origin='lower', vmin=0, vmax=1)
                axes_ch[0, 0].set_title(f'{channel_name} - Original', fontsize=12, fontweight='bold')
                axes_ch[0, 0].axis('off')
                plt.colorbar(im1, ax=axes_ch[0, 0], fraction=0.046, pad=0.04)
                
                # Row 1, Col 2: Magnitude
                im2 = axes_ch[0, 1].imshow(np.abs(Y), cmap='magma', origin='lower')
                axes_ch[0, 1].set_title(f'{channel_name} - Magnitude', fontsize=12, fontweight='bold')
                axes_ch[0, 1].axis('off')
                plt.colorbar(im2, ax=axes_ch[0, 1], fraction=0.046, pad=0.04)
                
                # Row 1, Col 3: Real Part
                im3 = axes_ch[0, 2].imshow(Y.real, cmap='viridis', origin='lower',
                                        vmin=np.percentile(Y.real, 2),
                                        vmax=np.percentile(Y.real, 98))
                axes_ch[0, 2].set_title(f'{channel_name} - Real Part', fontsize=12, fontweight='bold')
                axes_ch[0, 2].axis('off')
                plt.colorbar(im3, ax=axes_ch[0, 2], fraction=0.046, pad=0.04)
                
                # Row 2, Col 1: Imaginary Part
                imag_abs_max = np.percentile(np.abs(Y.imag), 98)
                im4 = axes_ch[1, 0].imshow(Y.imag, cmap='plasma', origin='lower',
                                        vmin=-imag_abs_max, vmax=imag_abs_max)
                axes_ch[1, 0].set_title(f'{channel_name} - Imaginary Part', fontsize=12, fontweight='bold')
                axes_ch[1, 0].axis('off')
                plt.colorbar(im4, ax=axes_ch[1, 0], fraction=0.046, pad=0.04)
                
                # Row 2, Col 2: Phase
                im5 = axes_ch[1, 1].imshow(np.angle(Y), cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
                axes_ch[1, 1].set_title(f'{channel_name} - Phase', fontsize=12, fontweight='bold')
                axes_ch[1, 1].axis('off')
                plt.colorbar(im5, ax=axes_ch[1, 1], fraction=0.046, pad=0.04)
                
                # Row 2, Col 3: Empty (or could add another metric)
                axes_ch[1, 2].axis('off')
            else:
                # Without imaginary lambda: only show original, magnitude, real
                fig_ch, axes_ch = plt.subplots(1, 3, figsize=(15, 5))
                fig_ch.suptitle(f'MDFKT Analysis - {channel_name} Channel: {Path(image_path).name}', 
                               fontsize=16, fontweight='bold', y=0.98)
                
                # Col 1: Original
                im1 = axes_ch[0].imshow(f_original, cmap='viridis', origin='lower', vmin=0, vmax=1)
                axes_ch[0].set_title(f'{channel_name} - Original', fontsize=12, fontweight='bold')
                axes_ch[0].axis('off')
                plt.colorbar(im1, ax=axes_ch[0], fraction=0.046, pad=0.04)
                
                # Col 2: Magnitude
                im2 = axes_ch[1].imshow(np.abs(Y), cmap='magma', origin='lower')
                axes_ch[1].set_title(f'{channel_name} - Magnitude', fontsize=12, fontweight='bold')
                axes_ch[1].axis('off')
                plt.colorbar(im2, ax=axes_ch[1], fraction=0.046, pad=0.04)
                
                # Col 3: Real Part
                im3 = axes_ch[2].imshow(Y.real, cmap='viridis', origin='lower',
                                        vmin=np.percentile(Y.real, 2),
                                        vmax=np.percentile(Y.real, 98))
                axes_ch[2].set_title(f'{channel_name} - Real Part', fontsize=12, fontweight='bold')
                axes_ch[2].axis('off')
                plt.colorbar(im3, ax=axes_ch[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save individual channel figure
            output_path_ch = Path(output_dir) / f"{Path(image_path).stem}_mdfkt_{channel_name}_channel.png"
            plt.savefig(output_path_ch, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"{channel_name} channel visualization saved to: {output_path_ch}")
            plt.show()
            plt.close()
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
        print("="*70)
        print(f"  📊 Figure 1: Full combined analysis")
        print(f"  📊 Figure 2: RGB composite views")
        print(f"  📊 Figure 3: R channel detailed analysis")
        print(f"  📊 Figure 4: G channel detailed analysis")
        print(f"  📊 Figure 5: B channel detailed analysis")
        print("="*70)

    def test_cosine_signal_with_lambda_variants(self, output_dir: Optional[str] = None):
        """
        Test how different Lambda eigenvalues transform a cosine test signal.
        Uses f(k) = cos(π·k/N) as input signal.
        Generates SEPARATE plots for each Lambda variant.
        
        Args:
            output_dir: Directory to save plots
        """
        k = np.arange(self.N)
        
        # Create test signal: cos(π·k/N)
        test_signal = np.cos(np.pi * k / self.N)
        
        print("\n" + "="*80)
        print("TESTING COSINE SIGNAL: f(k) = cos(π·k/N)")
        print("="*80)
        print(f"Signal length N = {self.N}")
        print(f"Signal range: [{test_signal.min():.6f}, {test_signal.max():.6f}]")
        print(f"First 10 values: {test_signal[:10]}")
        
        # Define Lambda variants to test
        lambda_variants = {
            'Reciprocal_1_over_k': 1.0 / (k + 1),
            'Inverse_Reciprocal_1_over_N_minus_k': 1.0 / (self.N - k),
            'Nth_Roots_of_Unity_plus1': np.exp(1j * 2 * np.pi * k / self.N),  # N-th roots of +1
            'Nth_Roots_of_Unity_minus1': np.exp(1j * np.pi * (2*k + 1) / self.N),  # N-th roots of -1
        }
        
        # Pretty names for titles
        lambda_pretty_names = {
            'Reciprocal_1_over_k': 'Reciprocal: λₖ = 1/k',
            'Inverse_Reciprocal_1_over_N_minus_k': 'Inverse Reciprocal: λₖ = 1/(N-k)',
            'Nth_Roots_of_Unity_plus1': 'N-th Roots of +1: λₖ = exp(i·2πk/N)',
            'Nth_Roots_of_Unity_minus1': 'N-th Roots of -1: λₖ = exp(i·π(2k+1)/N)',
        }
        
        # Store results
        results = {}
        
        # Setup output directory
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each Lambda variant SEPARATELY
        for lambda_key, Lambda in lambda_variants.items():
            lambda_name = lambda_pretty_names[lambda_key]
            print(f"\n{'='*80}")
            print(f"Processing: {lambda_name}")
            print(f"{'='*80}")
            
            # Apply transform: Y = Lambda * (K0 @ f)
            transformed = Lambda * (self.K0 @ test_signal)
            
            # Extract components
            real_part = np.real(transformed)
            imag_part = np.imag(transformed)
            magnitude = np.abs(transformed)
            phase = np.angle(transformed)
            
            # Check if Lambda is real or complex
            is_lambda_real = not np.iscomplexobj(Lambda)
            is_transformed_real = np.allclose(imag_part, 0, atol=1e-10)
            
            # Store results
            results[lambda_key] = {
                'lambda': Lambda,
                'transformed': transformed,
                'real': real_part,
                'imag': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'is_real': is_transformed_real
            }
            
            # Print statistics
            print(f"Lambda Type: {'Real' if is_lambda_real else 'Complex'}")
            print(f"Transformed Output: {'Real' if is_transformed_real else 'Complex'}")
            print(f"Transform Statistics:")
            print(f"  Real range: [{real_part.min():.6f}, {real_part.max():.6f}]")
            print(f"  Imag range: [{imag_part.min():.6f}, {imag_part.max():.6f}]")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Energy: {np.sum(magnitude**2):.6f}")
            
            # ========== CREATE SEPARATE FIGURE FOR THIS LAMBDA ==========
            # Determine number of subplots based on whether transformed is real or complex
            if is_transformed_real:
                # Real transformed: show only Input and Transformed (2 plots)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = cos(π·k/N), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                # Plot 1: Input Signal
                axes[0].plot(k, test_signal, 'k-', linewidth=2.5, label='Input Signal')
                axes[0].scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('k (sample index)', fontsize=11)
                axes[0].set_ylabel('Amplitude', fontsize=11)
                axes[0].set_title('Input: f(k) = cos(π·k/N)', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(fontsize=10)
                
                # Plot 2: Transformed Signal
                axes[1].plot(k, real_part, 'purple', linewidth=2.5, label='Y[k] (Transformed)')
                axes[1].scatter(k, real_part, c='purple', s=30, alpha=0.6, zorder=3)
                axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[1].set_xlabel('k (frequency index)', fontsize=11)
                axes[1].set_ylabel('Amplitude', fontsize=11)
                axes[1].set_title('Transformed Signal: Y[k]', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=10)
                
            else:
                # Complex transformed: Different layout - 2 plots top, 3 plots bottom
                fig = plt.figure(figsize=(18, 12))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = cos(π·k/N), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                # Create custom grid: 2 rows, first row has 2 plots, second row has 3 plots
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
                
                # Plot 1: Input Signal (top-left, spans 1.5 columns)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(k, test_signal, 'k-', linewidth=2.5, label='Input Signal')
                ax1.scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_xlabel('k (sample index)', fontsize=11)
                ax1.set_ylabel('Amplitude', fontsize=11)
                ax1.set_title('Input: f(k) = cos(π·k/N)', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=10)
                
                # Plot 2: Magnitude (top-right, spans 1.5 columns)
                ax2 = fig.add_subplot(gs[0, 1:])
                ax2.plot(k, magnitude, 'g-', linewidth=2.5, label='|Y[k]|')
                ax2.scatter(k, magnitude, c='green', s=30, alpha=0.6, zorder=3)
                ax2.set_xlabel('k (frequency index)', fontsize=11)
                ax2.set_ylabel('Magnitude', fontsize=11)
                ax2.set_title('Output Magnitude: |Y[k]|', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=10)
                
                # Plot 3: Real Part (bottom-left)
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.plot(k, real_part, 'b-', linewidth=2.5, label='Re{Y[k]}')
                ax3.scatter(k, real_part, c='blue', s=30, alpha=0.6, zorder=3)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_xlabel('k (frequency index)', fontsize=11)
                ax3.set_ylabel('Real Part', fontsize=11)
                ax3.set_title('Output Real Part: Re{Y[k]}', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=10)
                
                # Plot 4: Imaginary Part (bottom-middle)
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.plot(k, imag_part, 'r-', linewidth=2.5, label='Im{Y[k]}')
                ax4.scatter(k, imag_part, c='red', s=30, alpha=0.6, zorder=3)
                ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_xlabel('k (frequency index)', fontsize=11)
                ax4.set_ylabel('Imaginary Part', fontsize=11)
                ax4.set_title('Output Imaginary Part: Im{Y[k]}', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=10)
                
                # Plot 5: Phase (bottom-right)
                ax5 = fig.add_subplot(gs[1, 2])
                ax5.plot(k, phase, 'm-', linewidth=2.5, label='∠Y[k]')
                ax5.scatter(k, phase, c='magenta', s=30, alpha=0.6, zorder=3)
                ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax5.axhline(y=np.pi, color='gray', linestyle=':', alpha=0.5, label='±π')
                ax5.axhline(y=-np.pi, color='gray', linestyle=':', alpha=0.5)
                ax5.set_xlabel('k (frequency index)', fontsize=11)
                ax5.set_ylabel('Phase (radians)', fontsize=11)
                ax5.set_title('Output Phase: ∠Y[k]', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                ax5.legend(fontsize=10)
                ax5.set_ylim([-np.pi - 0.5, np.pi + 0.5])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save individual plot
            if output_dir:
                output_path = Path(output_dir) / f"cosine_test_{lambda_key}_N{self.N}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved: {output_path.name}")
            
            plt.show()
            plt.close()
        
        # Summary comparison
        print("\n" + "="*80)
        print("SUMMARY - ENERGY DISTRIBUTION COMPARISON")
        print("="*80)
        print(f"{'Lambda Method':<45} | {'Type':>10} | {'Energy':>12} | {'Max Magnitude':>14}")
        print("-"*80)
        for lambda_key, res in results.items():
            energy = np.sum(res['magnitude']**2)
            max_mag = np.max(res['magnitude'])
            signal_type = 'Real' if res['is_real'] else 'Complex'
            pretty_name = lambda_pretty_names[lambda_key]
            print(f"{pretty_name:<45} | {signal_type:>10} | {energy:>12.4f} | {max_mag:>14.4f}")
        print("="*80)
        
        return results
    

    def test_sine_signal_with_lambda_variants(self, output_dir: Optional[str] = None):
        """
        Test how different Lambda eigenvalues transform a sine test signal.
        Uses f(k) = sin(k·π/N) as input signal.
        Generates SEPARATE plots for each Lambda variant.
        
        Args:
            output_dir: Directory to save plots
        """
        k = np.arange(self.N)
        
        # Create test signal: sin(k·π/N)
        test_signal = np.sin(k * np.pi / self.N)
        
        print("\n" + "="*80)
        print("TESTING SINE SIGNAL: f(k) = sin(k·π/N)")
        print("="*80)
        print(f"Signal length N = {self.N}")
        print(f"Signal range: [{test_signal.min():.6f}, {test_signal.max():.6f}]")
        print(f"First 10 values: {test_signal[:10]}")
        
        # Define Lambda variants to test
        lambda_variants = {
            'Reciprocal_1_over_k': 1.0 / (k + 1),
            'Inverse_Reciprocal_1_over_N_minus_k': 1.0 / (self.N - k),
            'Nth_Roots_of_Unity_plus1': np.exp(1j * 2 * np.pi * k / self.N),
            'Nth_Roots_of_Unity_minus1': np.exp(1j * np.pi * (2*k + 1) / self.N),
        }
        
        lambda_pretty_names = {
            'Reciprocal_1_over_k': 'Reciprocal: λₖ = 1/k',
            'Inverse_Reciprocal_1_over_N_minus_k': 'Inverse Reciprocal: λₖ = 1/(N-k)',
            'Nth_Roots_of_Unity_plus1': 'N-th Roots of +1: λₖ = exp(i·2πk/N)',
            'Nth_Roots_of_Unity_minus1': 'N-th Roots of -1: λₖ = exp(i·π(2k+1)/N)',
        }
        
        results = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for lambda_key, Lambda in lambda_variants.items():
            lambda_name = lambda_pretty_names[lambda_key]
            print(f"\n{'='*80}")
            print(f"Processing: {lambda_name}")
            print(f"{'='*80}")
            
            transformed = Lambda * (self.K0 @ test_signal)
            
            real_part = np.real(transformed)
            imag_part = np.imag(transformed)
            magnitude = np.abs(transformed)
            phase = np.angle(transformed)
            
            is_lambda_real = not np.iscomplexobj(Lambda)
            is_transformed_real = np.allclose(imag_part, 0, atol=1e-10)
            
            results[lambda_key] = {
                'lambda': Lambda,
                'transformed': transformed,
                'real': real_part,
                'imag': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'is_real': is_transformed_real
            }
            
            print(f"Lambda Type: {'Real' if is_lambda_real else 'Complex'}")
            print(f"Transformed Output: {'Real' if is_transformed_real else 'Complex'}")
            print(f"Transform Statistics:")
            print(f"  Real range: [{real_part.min():.6f}, {real_part.max():.6f}]")
            print(f"  Imag range: [{imag_part.min():.6f}, {imag_part.max():.6f}]")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Energy: {np.sum(magnitude**2):.6f}")
            
            if is_transformed_real:
                # Real transformed: show only Input and Transformed (2 plots)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = sin(k·π/N), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                axes[0].plot(k, test_signal, 'k-', linewidth=2.5)
                axes[0].scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('k', fontsize=11)
                axes[0].set_ylabel('Amplitude', fontsize=11)
                axes[0].set_title('Input: f(k) = sin(k·π/N)', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(k, real_part, 'purple', linewidth=2.5)
                axes[1].scatter(k, real_part, c='purple', s=30, alpha=0.6, zorder=3)
                axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[1].set_xlabel('k', fontsize=11)
                axes[1].set_ylabel('Amplitude', fontsize=11)
                axes[1].set_title('Transformed: Y[k]', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
            else:
                fig = plt.figure(figsize=(18, 12))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = sin(k·π/N), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
                
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(k, test_signal, 'k-', linewidth=2.5)
                ax1.scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_xlabel('k', fontsize=11)
                ax1.set_ylabel('Amplitude', fontsize=11)
                ax1.set_title('Input: f(k) = sin(k·π/N)', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(gs[0, 1:])
                ax2.plot(k, magnitude, 'g-', linewidth=2.5)
                ax2.scatter(k, magnitude, c='green', s=30, alpha=0.6, zorder=3)
                ax2.set_xlabel('k', fontsize=11)
                ax2.set_ylabel('Magnitude', fontsize=11)
                ax2.set_title('Magnitude: |Y[k]|', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.plot(k, real_part, 'b-', linewidth=2.5)
                ax3.scatter(k, real_part, c='blue', s=30, alpha=0.6, zorder=3)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_xlabel('k', fontsize=11)
                ax3.set_ylabel('Real Part', fontsize=11)
                ax3.set_title('Real: Re{Y[k]}', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.plot(k, imag_part, 'r-', linewidth=2.5)
                ax4.scatter(k, imag_part, c='red', s=30, alpha=0.6, zorder=3)
                ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_xlabel('k', fontsize=11)
                ax4.set_ylabel('Imaginary Part', fontsize=11)
                ax4.set_title('Imaginary: Im{Y[k]}', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                ax5 = fig.add_subplot(gs[1, 2])
                ax5.plot(k, phase, 'm-', linewidth=2.5)
                ax5.scatter(k, phase, c='magenta', s=30, alpha=0.6, zorder=3)
                ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax5.set_xlabel('k', fontsize=11)
                ax5.set_ylabel('Phase (radians)', fontsize=11)
                ax5.set_title('Phase: ∠Y[k]', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                ax5.set_ylim([-np.pi - 0.5, np.pi + 0.5])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if output_dir:
                output_path = Path(output_dir) / f"sine_test_{lambda_key}_N{self.N}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved: {output_path.name}")
            
            plt.show()
            plt.close()
        
        print("\n" + "="*80)
        print("SUMMARY - SINE SIGNAL ENERGY COMPARISON")
        print("="*80)
        for lambda_key, res in results.items():
            energy = np.sum(res['magnitude']**2)
            signal_type = 'Real' if res['is_real'] else 'Complex'
            print(f"{lambda_pretty_names[lambda_key]:<45} | {signal_type:>10} | Energy: {energy:>12.4f}")
        print("="*80)
        
        return results

    def test_reciprocal_signal_with_lambda_variants(self, output_dir: Optional[str] = None):
        """
        Test how different Lambda eigenvalues transform a reciprocal test signal.
        Uses f(k) = 1/(k+1) as input signal (shifted to avoid division by zero).
        Generates SEPARATE plots for each Lambda variant.
        
        Args:
            output_dir: Directory to save plots
        """
        k = np.arange(self.N)
        
        # Create test signal: 1/(k+1) to avoid division by zero
        test_signal = 1.0 / (k + 1)
        
        print("\n" + "="*80)
        print("TESTING RECIPROCAL SIGNAL: f(k) = 1/(k+1)")
        print("="*80)
        print(f"Signal length N = {self.N}")
        print(f"Signal range: [{test_signal.min():.6f}, {test_signal.max():.6f}]")
        print(f"First 10 values: {test_signal[:10]}")
        
        lambda_variants = {
            'Reciprocal_1_over_k': 1.0 / (k + 1),
            'Inverse_Reciprocal_1_over_N_minus_k': 1.0 / (self.N - k),
            'Nth_Roots_of_Unity_plus1': np.exp(1j * 2 * np.pi * k / self.N),
            'Nth_Roots_of_Unity_minus1': np.exp(1j * np.pi * (2*k + 1) / self.N),
        }
        
        lambda_pretty_names = {
            'Reciprocal_1_over_k': 'Reciprocal: λₖ = 1/k',
            'Inverse_Reciprocal_1_over_N_minus_k': 'Inverse Reciprocal: λₖ = 1/(N-k)',
            'Nth_Roots_of_Unity_plus1': 'N-th Roots of +1: λₖ = exp(i·2πk/N)',
            'Nth_Roots_of_Unity_minus1': 'N-th Roots of -1: λₖ = exp(i·π(2k+1)/N)',
        }
        
        results = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for lambda_key, Lambda in lambda_variants.items():
            lambda_name = lambda_pretty_names[lambda_key]
            print(f"\n{'='*80}")
            print(f"Processing: {lambda_name}")
            print(f"{'='*80}")
            
            transformed = Lambda * (self.K0 @ test_signal)
            
            real_part = np.real(transformed)
            imag_part = np.imag(transformed)
            magnitude = np.abs(transformed)
            phase = np.angle(transformed)
            
            is_lambda_real = not np.iscomplexobj(Lambda)
            is_transformed_real = np.allclose(imag_part, 0, atol=1e-10)
            
            results[lambda_key] = {
                'lambda': Lambda,
                'transformed': transformed,
                'real': real_part,
                'imag': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'is_real': is_transformed_real
            }
            
            print(f"Lambda Type: {'Real' if is_lambda_real else 'Complex'}")
            print(f"Transformed Output: {'Real' if is_transformed_real else 'Complex'}")
            print(f"Transform Statistics:")
            print(f"  Real range: [{real_part.min():.6f}, {real_part.max():.6f}]")
            print(f"  Imag range: [{imag_part.min():.6f}, {imag_part.max():.6f}]")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Energy: {np.sum(magnitude**2):.6f}")
            
            if is_transformed_real:
                # Real transformed: show only Input and Transformed (2 plots)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = 1/(k+1), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                axes[0].plot(k, test_signal, 'k-', linewidth=2.5)
                axes[0].scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                axes[0].set_xlabel('k', fontsize=11)
                axes[0].set_ylabel('Amplitude', fontsize=11)
                axes[0].set_title('Input: f(k) = 1/(k+1)', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(k, real_part, 'purple', linewidth=2.5)
                axes[1].scatter(k, real_part, c='purple', s=30, alpha=0.6, zorder=3)
                axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[1].set_xlabel('k', fontsize=11)
                axes[1].set_ylabel('Amplitude', fontsize=11)
                axes[1].set_title('Transformed: Y[k]', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
            else:
                fig = plt.figure(figsize=(18, 12))
                fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = 1/(k+1), N={self.N}', 
                            fontsize=14, fontweight='bold')
                
                gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
                
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(k, test_signal, 'k-', linewidth=2.5)
                ax1.scatter(k, test_signal, c='black', s=30, alpha=0.6, zorder=3)
                ax1.set_xlabel('k', fontsize=11)
                ax1.set_ylabel('Amplitude', fontsize=11)
                ax1.set_title('Input: f(k) = 1/(k+1)', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                ax2 = fig.add_subplot(gs[0, 1:])
                ax2.plot(k, magnitude, 'g-', linewidth=2.5)
                ax2.scatter(k, magnitude, c='green', s=30, alpha=0.6, zorder=3)
                ax2.set_xlabel('k', fontsize=11)
                ax2.set_ylabel('Magnitude', fontsize=11)
                ax2.set_title('Magnitude: |Y[k]|', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.plot(k, real_part, 'b-', linewidth=2.5)
                ax3.scatter(k, real_part, c='blue', s=30, alpha=0.6, zorder=3)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_xlabel('k', fontsize=11)
                ax3.set_ylabel('Real Part', fontsize=11)
                ax3.set_title('Real: Re{Y[k]}', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.plot(k, imag_part, 'r-', linewidth=2.5)
                ax4.scatter(k, imag_part, c='red', s=30, alpha=0.6, zorder=3)
                ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_xlabel('k', fontsize=11)
                ax4.set_ylabel('Imaginary Part', fontsize=11)
                ax4.set_title('Imaginary: Im{Y[k]}', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                ax5 = fig.add_subplot(gs[1, 2])
                ax5.plot(k, phase, 'm-', linewidth=2.5)
                ax5.scatter(k, phase, c='magenta', s=30, alpha=0.6, zorder=3)
                ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax5.set_xlabel('k', fontsize=11)
                ax5.set_ylabel('Phase (radians)', fontsize=11)
                ax5.set_title('Phase: ∠Y[k]', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                ax5.set_ylim([-np.pi - 0.5, np.pi + 0.5])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if output_dir:
                output_path = Path(output_dir) / f"reciprocal_test_{lambda_key}_N{self.N}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved: {output_path.name}")
            
            plt.show()
            plt.close()
        
        print("\n" + "="*80)
        print("SUMMARY - RECIPROCAL SIGNAL ENERGY COMPARISON")
        print("="*80)
        for lambda_key, res in results.items():
            energy = np.sum(res['magnitude']**2)
            signal_type = 'Real' if res['is_real'] else 'Complex'
            print(f"{lambda_pretty_names[lambda_key]:<45} | {signal_type:>10} | Energy: {energy:>12.4f}")
        print("="*80)
        
        return results


    def test_complex_exponential_signal_with_lambda_variants(self, output_dir: Optional[str] = None):
        """
        Test how different Lambda eigenvalues transform a complex exponential signal.
        Uses f(k) = exp(i·k·π/N) as input signal.
        Generates SEPARATE plots for each Lambda variant.
        
        Args:
            output_dir: Directory to save plots
        """
        k = np.arange(self.N)
        
        # Create test signal: exp(i·k·π/N)
        test_signal = np.exp(1j * k * np.pi / self.N)
        
        print("\n" + "="*80)
        print("TESTING COMPLEX EXPONENTIAL SIGNAL: f(k) = exp(i·k·π/N)")
        print("="*80)
        print(f"Signal length N = {self.N}")
        print(f"Signal type: Complex")
        print(f"Magnitude range: [{np.abs(test_signal).min():.6f}, {np.abs(test_signal).max():.6f}]")
        print(f"First 5 values: {test_signal[:5]}")
        
        lambda_variants = {
            'Reciprocal_1_over_k': 1.0 / (k + 1),
            'Inverse_Reciprocal_1_over_N_minus_k': 1.0 / (self.N - k),
            'Nth_Roots_of_Unity_plus1': np.exp(1j * 2 * np.pi * k / self.N),
            'Nth_Roots_of_Unity_minus1': np.exp(1j * np.pi * (2*k + 1) / self.N),
        }
        
        lambda_pretty_names = {
            'Reciprocal_1_over_k': 'Reciprocal: λₖ = 1/k',
            'Inverse_Reciprocal_1_over_N_minus_k': 'Inverse Reciprocal: λₖ = 1/(N-k)',
            'Nth_Roots_of_Unity_plus1': 'N-th Roots of +1: λₖ = exp(i·2πk/N)',
            'Nth_Roots_of_Unity_minus1': 'N-th Roots of -1: λₖ = exp(i·π(2k+1)/N)',
        }
        
        results = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for lambda_key, Lambda in lambda_variants.items():
            lambda_name = lambda_pretty_names[lambda_key]
            print(f"\n{'='*80}")
            print(f"Processing: {lambda_name}")
            print(f"{'='*80}")
            
            transformed = Lambda * (self.K0 @ test_signal)
            
            real_part = np.real(transformed)
            imag_part = np.imag(transformed)
            magnitude = np.abs(transformed)
            phase = np.angle(transformed)
            
            is_lambda_real = not np.iscomplexobj(Lambda)
            is_transformed_real = np.allclose(imag_part, 0, atol=1e-10)
            
            results[lambda_key] = {
                'lambda': Lambda,
                'transformed': transformed,
                'real': real_part,
                'imag': imag_part,
                'magnitude': magnitude,
                'phase': phase,
                'is_real': is_transformed_real
            }
            
            print(f"Lambda Type: {'Real' if is_lambda_real else 'Complex'}")
            print(f"Transformed Output: {'Real' if is_transformed_real else 'Complex'}")
            print(f"Transform Statistics:")
            print(f"  Real range: [{real_part.min():.6f}, {real_part.max():.6f}]")
            print(f"  Imag range: [{imag_part.min():.6f}, {imag_part.max():.6f}]")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Energy: {np.sum(magnitude**2):.6f}")
            
            # Always use complex layout since input is complex
            fig = plt.figure(figsize=(18, 12))
            fig.suptitle(f'{lambda_name}\nTest Signal: f(k) = exp(i·k·π/N), N={self.N}', 
                        fontsize=14, fontweight='bold')
            
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Input Signal Magnitude
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(k, np.abs(test_signal), 'k-', linewidth=2.5, label='|f(k)|')
            ax1.scatter(k, np.abs(test_signal), c='black', s=30, alpha=0.6, zorder=3)
            ax1.set_xlabel('k', fontsize=11)
            ax1.set_ylabel('Magnitude', fontsize=11)
            ax1.set_title('Input Magnitude: |f(k)| = 1', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Plot 2: Output Magnitude
            ax2 = fig.add_subplot(gs[0, 1:])
            ax2.plot(k, magnitude, 'g-', linewidth=2.5, label='|Y[k]|')
            ax2.scatter(k, magnitude, c='green', s=30, alpha=0.6, zorder=3)
            ax2.set_xlabel('k', fontsize=11)
            ax2.set_ylabel('Magnitude', fontsize=11)
            ax2.set_title('Output Magnitude: |Y[k]|', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Plot 3: Real Part
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(k, real_part, 'b-', linewidth=2.5, label='Re{Y[k]}')
            ax3.scatter(k, real_part, c='blue', s=30, alpha=0.6, zorder=3)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('k', fontsize=11)
            ax3.set_ylabel('Real Part', fontsize=11)
            ax3.set_title('Real: Re{Y[k]}', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            
            # Plot 4: Imaginary Part
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(k, imag_part, 'r-', linewidth=2.5, label='Im{Y[k]}')
            ax4.scatter(k, imag_part, c='red', s=30, alpha=0.6, zorder=3)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('k', fontsize=11)
            ax4.set_ylabel('Imaginary Part', fontsize=11)
            ax4.set_title('Imaginary: Im{Y[k]}', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
            
            # Plot 5: Phase
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.plot(k, phase, 'm-', linewidth=2.5, label='∠Y[k]')
            ax5.scatter(k, phase, c='magenta', s=30, alpha=0.6, zorder=3)
            ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax5.axhline(y=np.pi, color='gray', linestyle=':', alpha=0.5)
            ax5.axhline(y=-np.pi, color='gray', linestyle=':', alpha=0.5)
            ax5.set_xlabel('k', fontsize=11)
            ax5.set_ylabel('Phase (radians)', fontsize=11)
            ax5.set_title('Phase: ∠Y[k]', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=10)
            ax5.set_ylim([-np.pi - 0.5, np.pi + 0.5])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if output_dir:
                output_path = Path(output_dir) / f"complex_exp_test_{lambda_key}_N{self.N}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✓ Saved: {output_path.name}")
            
            plt.show()
            plt.close()
        
        print("\n" + "="*80)
        print("SUMMARY - COMPLEX EXPONENTIAL ENERGY COMPARISON")
        print("="*80)
        for lambda_key, res in results.items():
            energy = np.sum(res['magnitude']**2)
            signal_type = 'Real' if res['is_real'] else 'Complex'
            print(f"{lambda_pretty_names[lambda_key]:<45} | {signal_type:>10} | Energy: {energy:>12.4f}")
        print("="*80)
        
        return results

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MDFKTImageAnalyzer(N=64, p=0.5)
    

    # Analyze a bcc/bkl image
    image_path = "ISIC_0024885_segmented_original.png"  

    """
    # # Test cosine signal with different Lambda variants
    # print("\n" + "="*70)
    # print("TESTING: f(k) = cos(π·k/N) with Different Lambda Eigenvalues")
    # print("="*70)
    #results = analyzer.test_cosine_signal_with_lambda_variants(output_dir="mdfkt_cos_results")

    #results = analyzer.test_sine_signal_with_lambda_variants(output_dir="mdfkt_sin_results")

    #results = analyzer.test_reciprocal_signal_with_lambda_variants(output_dir="mdfkt_reciprocal_results")

    #results = analyzer.test_complex_exponential_signal_with_lambda_variants(output_dir="mdfkt_complex_exp_results")




    
    """
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    
    try:

        results = analyzer.test_cosine_signal_with_lambda_variants(output_dir="mdfkt_cos_results")

        results = analyzer.test_sine_signal_with_lambda_variants(output_dir="mdfkt_sin_results")

        results = analyzer.test_reciprocal_signal_with_lambda_variants(output_dir="mdfkt_reciprocal_results")

        results = analyzer.test_complex_exponential_signal_with_lambda_variants(output_dir="mdfkt_complex_exp_results")
        # results = analyzer.analyze_image(
        #     image_path=image_path,
        #     output_dir=None  # Will create 'mdfkt_results' folder automatically
        # )
        
        # print("\n" + "="*70)
        # print("ANALYSIS COMPLETE!")
        # print("="*70)
        # print(f"Total features extracted: {len(results['all_features'])}")
        # print(f"Check the output folder for visualizations and CSV file.")
  
        
    except FileNotFoundError:
        print(f"\n⚠️  Error: Image file not found!")
        print(f"Please update the 'image_path' variable with a valid melanoma image path.")
        print(f"Current path: {image_path}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

