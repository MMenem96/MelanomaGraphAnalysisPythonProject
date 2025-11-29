import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import color
from scipy.special import comb
import logging

def visualize_krawtchouk_analysis(image_path, output_path='krawtchouk_analysis_figure.png', max_order=4, dpi=300):
    """
    Create a comprehensive 6-panel figure demonstrating Krawtchouk moment analysis.
    
    This function generates a publication-ready figure showing:
    1. Original segmented lesion image
    2. Krawtchouk moment magnitude heatmap
    3. Krawtchouk moment phase heatmap
    4. Real part of moments heatmap
    5. Imaginary part of moments heatmap
    6. Reconstructed image from moments
    
    Args:
        image_path: Path to segmented lesion image (e.g., 'data/bcc_segmented/image.png')
        output_path: Path to save the output figure
        max_order: Maximum Krawtchouk moment order (default: 4)
        dpi: Resolution for saved figure (default: 300)
    """
    
    # ============ HELPER FUNCTIONS ============
    
    def krawtchouk_polynomial(x, n, p, N):
        """Calculate Krawtchouk polynomial value."""
        if n > N or x > N or x < 0:
            return 0.0
        
        result = 0.0
        for k in range(n + 1):
            if x >= k and (N - x) >= (n - k):
                binom_x = comb(x, k, exact=True)
                binom_N = comb(N - x, n - k, exact=True)
                a_k = ((-1) ** k) * comb(n, k, exact=True) * (p ** (n - k)) * ((1 - p) ** k)
                result += a_k * binom_x * binom_N
        
        return result
    
    def compute_krawtchouk_moments(image, max_order=4, p1=0.5, p2=0.5):
        """Compute Krawtchouk moments up to max_order."""
        N1, N2 = image.shape
        
        # Pre-compute Krawtchouk polynomials
        K_x = np.zeros((max_order + 1, N2))
        K_y = np.zeros((max_order + 1, N1))
        
        for n in range(max_order + 1):
            for x in range(N2):
                K_x[n, x] = krawtchouk_polynomial(x, n, p1, N2 - 1)
            for y in range(N1):
                K_y[n, y] = krawtchouk_polynomial(y, n, p2, N1 - 1)
        
        # Compute moments
        moments = np.zeros((max_order + 1, max_order + 1), dtype=complex)
        
        for n in range(max_order + 1):
            for m in range(max_order + 1):
                if n + m <= max_order:
                    moment = 0.0 + 0.0j
                    for y in range(N1):
                        for x in range(N2):
                            moment += image[y, x] * K_x[n, x] * K_y[m, y]
                    
                    # Normalize
                    moments[n, m] = moment / (N1 * N2)
        
        return moments, K_x, K_y
    
    def reconstruct_from_moments(moments, K_x, K_y, shape):
        """Reconstruct image from Krawtchouk moments."""
        N1, N2 = shape
        max_order = moments.shape[0] - 1
        
        reconstructed = np.zeros((N1, N2))
        
        for n in range(max_order + 1):
            for m in range(max_order + 1):
                if n + m <= max_order:
                    for y in range(N1):
                        for x in range(N2):
                            reconstructed[y, x] += np.real(moments[n, m] * K_x[n, x] * K_y[m, y])
        
        # Normalize to [0, 1]
        reconstructed = np.clip(reconstructed, 0, 1)
        
        return reconstructed
    
    # ============ MAIN PROCESSING ============
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image.copy()
        
        # Create mask (non-black pixels)
        mask = np.sum(image, axis=2) > 10 if len(image.shape) == 3 else image > 0.01
        
        # Extract lesion region
        rows, cols = np.where(mask)
        if len(rows) == 0:
            raise ValueError("No lesion found in image (all black)")
        
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        lesion_region = gray[min_row:max_row+1, min_col:max_col+1].copy()
        lesion_mask = mask[min_row:max_row+1, min_col:max_col+1]
        lesion_region[~lesion_mask] = 0
        
        # Resize for computational efficiency
        normalize_size = 64
        if max(lesion_region.shape) > normalize_size:
            scale = normalize_size / max(lesion_region.shape)
            new_height = int(lesion_region.shape[0] * scale)
            new_width = int(lesion_region.shape[1] * scale)
            lesion_region = cv2.resize(lesion_region, (new_width, new_height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Compute Krawtchouk moments
        print("Computing Krawtchouk moments...")
        moments, K_x, K_y = compute_krawtchouk_moments(lesion_region, max_order=max_order)
        
        # Reconstruct image
        print("Reconstructing image from moments...")
        reconstructed = reconstruct_from_moments(moments, K_x, K_y, lesion_region.shape)
        
        # Extract moment components
        magnitude = np.abs(moments)
        phase = np.angle(moments)
        real_part = np.real(moments)
        imag_part = np.imag(moments)
        
        # ============ CREATE FIGURE ============
        
        print("Creating visualization...")
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Original Segmented Image
        ax1 = fig.add_subplot(gs[0, 0])
        original_display = image[min_row:max_row+1, min_col:max_col+1]
        ax1.imshow(original_display)
        ax1.set_title('(a) Original Segmented Lesion', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Magnitude Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(magnitude, cmap='hot', interpolation='nearest')
        ax2.set_title('(b) Moment Magnitude |Q_nm|', fontsize=12, fontweight='bold')
        ax2.set_xlabel('m (horizontal order)')
        ax2.set_ylabel('n (vertical order)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Add grid
        ax2.set_xticks(range(max_order + 1))
        ax2.set_yticks(range(max_order + 1))
        ax2.grid(True, alpha=0.3)
        
        # 3. Phase Heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(phase, cmap='twilight', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
        ax3.set_title('(c) Moment Phase ∠Q_nm', fontsize=12, fontweight='bold')
        ax3.set_xlabel('m (horizontal order)')
        ax3.set_ylabel('n (vertical order)')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Radians')
        ax3.set_xticks(range(max_order + 1))
        ax3.set_yticks(range(max_order + 1))
        ax3.grid(True, alpha=0.3)
        
        # 4. Real Part
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(real_part, cmap='RdBu_r', interpolation='nearest')
        ax4.set_title('(d) Real Part Re(Q_nm)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('m (horizontal order)')
        ax4.set_ylabel('n (vertical order)')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_xticks(range(max_order + 1))
        ax4.set_yticks(range(max_order + 1))
        ax4.grid(True, alpha=0.3)
        
        # 5. Imaginary Part
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(imag_part, cmap='RdBu_r', interpolation='nearest')
        ax5.set_title('(e) Imaginary Part Im(Q_nm)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('m (horizontal order)')
        ax5.set_ylabel('n (vertical order)')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_xticks(range(max_order + 1))
        ax5.set_yticks(range(max_order + 1))
        ax5.grid(True, alpha=0.3)
        
        # 6. Reconstructed Image
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(reconstructed, cmap='gray')
        ax6.set_title('(f) Reconstructed from Moments', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Add overall title
        fig.suptitle('Krawtchouk Moment Analysis for Skin Lesion Characterization', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {output_path}")
        
        # Also save as PDF for paper
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"PDF version saved to: {pdf_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("KRAWTCHOUK MOMENT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Image size: {lesion_region.shape}")
        print(f"Maximum moment order: {max_order}")
        print(f"Total moments computed: {np.sum(np.triu(np.ones((max_order+1, max_order+1))))}")
        print(f"\nMoment statistics:")
        print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
        print(f"  Mean magnitude: {magnitude.mean():.6f}")
        print(f"  Strongest moment: Q_{np.unravel_index(magnitude.argmax(), magnitude.shape)}")
        print(f"\nReconstruction quality:")
        print(f"  MSE: {np.mean((lesion_region - reconstructed)**2):.6f}")
        print(f"  PSNR: {10 * np.log10(1.0 / np.mean((lesion_region - reconstructed)**2)):.2f} dB")
        print("="*60)
        
        return fig, moments, reconstructed
        
    except Exception as e:
        logging.error(f"Error in Krawtchouk visualization: {str(e)}")
        raise


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Example: Analyze a BCC lesion
    image_path = "data/bcc_segmented/ISIC_0033579_segmented.png"  # Replace with your actual image path
    
    # Create the figure
    fig, moments, reconstructed = visualize_krawtchouk_analysis(
        image_path=image_path,
        output_path="krawtchouk_analysis_figure.png",
        max_order=4,
        dpi=300
    )