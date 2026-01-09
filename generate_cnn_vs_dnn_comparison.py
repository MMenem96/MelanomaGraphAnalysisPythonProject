"""
Visual Comparison: CNN vs DNN Architecture
Creates side-by-side diagrams showing the difference
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

fig = plt.figure(figsize=(16, 10))

# Colors
color_cnn = '#E3F2FD'
color_dnn = '#FFF3E0'
color_input = '#E8F5E9'
color_output = '#FCE4EC'

def draw_box(ax, x, y, width, height, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           color='black',
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=7, color='red', fontweight='bold')

# ============================================================================
# LEFT SIDE: CNN (NOT your system)
# ============================================================================
ax_cnn = fig.add_subplot(1, 2, 1)
ax_cnn.set_xlim(0, 10)
ax_cnn.set_ylim(0, 12)
ax_cnn.axis('off')

ax_cnn.text(5, 11.5, 'CNN Architecture', ha='center', fontsize=14, fontweight='bold', color='red')
ax_cnn.text(5, 11, '(NOT what you have)', ha='center', fontsize=11, color='red', style='italic')

# Input
draw_box(ax_cnn, 2, 9.5, 6, 0.8, 'Raw Image\n224×224×3 = 150,528 pixels', color_input, fontsize=10)
draw_arrow(ax_cnn, 5, 9.5, 5, 9)

# Convolution 1
draw_box(ax_cnn, 1.5, 8, 7, 0.8, 'Convolution Layer 1 (32 filters, 3×3)\nLearns edges, lines, simple patterns', color_cnn)
draw_arrow(ax_cnn, 5, 8, 5, 7.5)

# Pooling 1
draw_box(ax_cnn, 2, 6.8, 6, 0.6, 'Max Pooling (2×2)\nReduce size by half', '#BBDEFB', fontsize=8)
draw_arrow(ax_cnn, 5, 6.8, 5, 6.3)

# Convolution 2
draw_box(ax_cnn, 1.5, 5.3, 7, 0.8, 'Convolution Layer 2 (64 filters, 3×3)\nLearns textures, shapes', color_cnn)
draw_arrow(ax_cnn, 5, 5.3, 5, 4.8)

# Pooling 2
draw_box(ax_cnn, 2, 4.1, 6, 0.6, 'Max Pooling (2×2)\nReduce size again', '#BBDEFB', fontsize=8)
draw_arrow(ax_cnn, 5, 4.1, 5, 3.6)

# Convolution 3
draw_box(ax_cnn, 1.5, 2.6, 7, 0.8, 'Convolution Layer 3 (128 filters, 3×3)\nLearns complex patterns', color_cnn)
draw_arrow(ax_cnn, 5, 2.6, 5, 2.1)

# Flatten
draw_box(ax_cnn, 2, 1.4, 6, 0.6, 'Flatten\nConvert 3D → 1D vector', '#E1BEE7', fontsize=8)
draw_arrow(ax_cnn, 5, 1.4, 5, 0.9)

# Dense layers
draw_box(ax_cnn, 2.5, 0.2, 5, 0.6, 'Fully Connected Layers → Output', color_output)

# Annotation
ax_cnn.text(5, -0.8, 'Learns features AUTOMATICALLY from pixels', 
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# RIGHT SIDE: DNN (YOUR system)
# ============================================================================
ax_dnn = fig.add_subplot(1, 2, 2)
ax_dnn.set_xlim(0, 10)
ax_dnn.set_ylim(0, 12)
ax_dnn.axis('off')

ax_dnn.text(5, 11.5, 'DNN Architecture', ha='center', fontsize=14, fontweight='bold', color='green')
ax_dnn.text(5, 11, '(What YOU have!) ✅', ha='center', fontsize=11, color='green', style='italic')

# Input
draw_box(ax_dnn, 2, 9.5, 6, 0.8, 'Handcrafted Features\n360 numbers (after selection)', color_input, fontsize=10)
draw_arrow(ax_dnn, 5, 9.5, 5, 9, '360 features')

# Feature examples
ax_dnn.text(1, 8.8, 'Examples:', fontsize=8, fontweight='bold')
ax_dnn.text(1, 8.5, '• Border irregularity: 0.78', fontsize=7)
ax_dnn.text(1, 8.2, '• Color variance: 0.65', fontsize=7)
ax_dnn.text(1, 7.9, '• Asymmetry score: 0.82', fontsize=7)
ax_dnn.text(1, 7.6, '• GLCM contrast: 45.3', fontsize=7)
ax_dnn.text(1, 7.3, '• ... (356 more)', fontsize=7)

# Embedding
draw_box(ax_dnn, 1.5, 6.3, 7, 0.8, 'Embedding Layer\nDense(512) + BatchNorm + Swish + Dropout(0.3)', color_dnn)
draw_arrow(ax_dnn, 5, 6.3, 5, 5.8, '512 units')

# Residual blocks
draw_box(ax_dnn, 1.5, 4.8, 7, 0.8, 'Residual Blocks 1-2\n512 units, dropout=0.30', color_dnn)
ax_dnn.plot([8.6, 9, 9, 8.6], [5.2, 5.2, 4.8, 4.8], 'r-', linewidth=2)
ax_dnn.text(9.2, 5, 'Skip', fontsize=6, color='red')
draw_arrow(ax_dnn, 5, 4.8, 5, 4.3, '512 units')

draw_box(ax_dnn, 1.5, 3.3, 7, 0.8, 'Residual Blocks 3-4\n384 units, dropout=0.28', color_dnn)
ax_dnn.plot([8.6, 9, 9, 8.6], [3.7, 3.7, 3.3, 3.3], 'r-', linewidth=2)
draw_arrow(ax_dnn, 5, 3.3, 5, 2.8, '384 units')

draw_box(ax_dnn, 1.5, 1.8, 7, 0.8, 'Residual Blocks 5-7\n256 → 128 units', color_dnn)
ax_dnn.plot([8.6, 9, 9, 8.6], [2.2, 2.2, 1.8, 1.8], 'r-', linewidth=2)
draw_arrow(ax_dnn, 5, 1.8, 5, 1.3, '128 units')

# Attention
draw_box(ax_dnn, 2, 0.6, 6, 0.6, 'Self-Attention\nLearn feature importance', '#FCE4EC', fontsize=9)
draw_arrow(ax_dnn, 5, 0.6, 5, 0.1)

# Output
draw_box(ax_dnn, 2.5, -0.5, 5, 0.6, 'Output Layer → BCC vs SK', color_output)

# Annotation
ax_dnn.text(5, -1.3, 'Learns feature COMBINATIONS from pre-computed features', 
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============================================================================
# Bottom comparison table
# ============================================================================
ax_table = fig.add_axes([0.1, 0.02, 0.8, 0.12])
ax_table.axis('off')

comparison_text = """
┌─────────────────────┬─────────────────────────────────────┬─────────────────────────────────────┐
│                     │  CNN (Convolutional NN)             │  DNN (Deep Dense NN) ✅ YOUR SYSTEM │
├─────────────────────┼─────────────────────────────────────┼─────────────────────────────────────┤
│ Input Type          │  Raw pixels (images)                │  Numbers (features)                  │
│ Input Size          │  224×224×3 = 150,528 values         │  360 features                        │
│ Key Operation       │  Convolution (filters slide)        │  Fully Connected (all-to-all)        │
│ Learns              │  Visual features automatically      │  Feature combinations                │
│ Interpretability    │  ❌ Black box (hard to explain)     │  ✅ Can see which features matter   │
│ Data Needed         │  10,000+ images                     │  1,000+ samples (you have ~2,257)   │
│ Training Time       │  Hours on GPU                       │  Minutes on CPU                      │
│ Best For            │  When no features available         │  When good features exist (99% AUC!) │
└─────────────────────┴─────────────────────────────────────┴─────────────────────────────────────┘

🎯 YOUR CHOICE (DNN) IS CORRECT because:
   1. You already have excellent handcrafted features (SVM = 99% AUC)
   2. Clinical interpretability is important (doctors understand features)
   3. Limited medical data (~2,257 samples)
   4. 20-50× faster training than CNN
   5. Modern architecture (residual + attention) on proven features = Best of both worlds! 💡
"""

ax_table.text(0.5, 0.5, comparison_text,
             transform=ax_table.transAxes,
             fontsize=8, family='monospace',
             verticalalignment='center',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save figure
output_path = 'output/figures/cnn_vs_dnn_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Comparison diagram saved to: {output_path}")

output_path_pdf = 'output/figures/cnn_vs_dnn_comparison.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ PDF version saved to: {output_path_pdf}")

plt.show()

print("\n" + "="*70)
print("CNN vs DNN comparison figure generated!")
print("="*70)
print("\nKey Difference:")
print("• CNN: Learns from PIXELS (raw images)")
print("• DNN: Learns from FEATURES (your 360 handcrafted features)")
print("\nYour choice (DNN) is CORRECT! ✅")
print("="*70)
