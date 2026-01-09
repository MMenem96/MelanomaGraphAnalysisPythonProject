"""
Generate CNN Architecture Figure for Academic Paper
Creates a professional multi-panel diagram showing the complete melanoma classification pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set professional style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.5

# Create figure with specific layout
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3, 
                      left=0.05, right=0.95, top=0.93, bottom=0.05)

# Color scheme
color_input = '#E8F4F8'
color_handcrafted = '#FFF3E0'
color_dnn = '#E8EAF6'
color_attention = '#FCE4EC'
color_output = '#E8F5E9'
color_arrow = '#424242'

def draw_box(ax, x, y, width, height, text, color, fontsize=8, fontweight='normal'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5,
                         transform=ax.transData)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->'):
    """Draw arrow between components"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

def draw_residual_block(ax, x, y, width, height, units, dropout, block_num):
    """Draw a single residual block with skip connection"""
    # Main path
    block_height = height / 6
    
    # Dense 1
    draw_box(ax, x, y + 5*block_height, width*0.9, block_height*0.8,
            f'Dense({units})', '#BBDEFB', fontsize=7)
    
    # BatchNorm
    draw_box(ax, x, y + 4*block_height, width*0.9, block_height*0.8,
            'BatchNorm', '#B3E5FC', fontsize=7)
    
    # Activation
    draw_box(ax, x, y + 3*block_height, width*0.9, block_height*0.8,
            'Swish', '#81D4FA', fontsize=7)
    
    # Dropout
    draw_box(ax, x, y + 2*block_height, width*0.9, block_height*0.8,
            f'Dropout({dropout:.1f})', '#4FC3F7', fontsize=7)
    
    # Dense 2
    draw_box(ax, x, y + 1*block_height, width*0.9, block_height*0.8,
            f'Dense({units})', '#29B6F6', fontsize=7)
    
    # BatchNorm 2
    draw_box(ax, x, y, width*0.9, block_height*0.8,
            'BatchNorm', '#03A9F4', fontsize=7)
    
    # Skip connection (curved arrow on the right)
    skip_x = x + width*0.95
    ax.annotate('', xy=(skip_x, y + block_height*0.4), 
               xytext=(skip_x, y + 5.5*block_height),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=.3"))
    
    # Add + symbol
    ax.text(skip_x + 0.1, y + 2.5*block_height, '+',
           fontsize=12, fontweight='bold', color='red',
           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red'))
    
    # Block label
    ax.text(x + width/2, y + height + 0.1, f'Block {block_num}',
           ha='center', fontsize=7, fontweight='bold')

# ============================================================================
# MAIN ARCHITECTURE DIAGRAM
# ============================================================================

ax_main = fig.add_subplot(gs[:, :])
ax_main.set_xlim(0, 10)
ax_main.set_ylim(0, 10)
ax_main.axis('off')

# Title
ax_main.text(5, 9.5, 'Deep Neural Network Architecture for BCC vs SK Classification',
            ha='center', fontsize=14, fontweight='bold')

# ============================================================================
# 1. INPUT LAYER (Top)
# ============================================================================
input_y = 8.5
draw_box(ax_main, 4.2, input_y, 1.6, 0.6, 
        'Dermoscopic Image\n224×224×3', color_input, fontsize=9, fontweight='bold')

# Arrow down
draw_arrow(ax_main, 5, input_y, 5, 7.8)

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
prep_y = 7.5
draw_box(ax_main, 4.2, prep_y, 1.6, 0.5,
        'Preprocessing\nResize, Normalize', color_input, fontsize=8)

# Split into two branches
draw_arrow(ax_main, 5, prep_y, 2, 6.8)  # Left to handcrafted
draw_arrow(ax_main, 5, prep_y, 8, 6.8)  # Right to DNN

# ============================================================================
# 3. LEFT BRANCH - HANDCRAFTED FEATURES
# ============================================================================
hc_x = 0.5
hc_y = 6.5

ax_main.text(2, hc_y + 0.7, 'Handcrafted Feature Extraction',
            ha='center', fontsize=10, fontweight='bold')

# Feature extraction modules
modules = [
    ('Geometric\nShape, Border, Asymmetry\nCompactness, Convexity', 'N=50'),
    ('Texture\nGLCM, LBP, Gabor\nWavelets', 'N=144'),
    ('Color\nRGB, HSV, LAB\nStatistics', 'N=150'),
    ('MDFKT/Krawtchouk\nMoment Transform', 'N=167')
]

module_height = 0.8
for i, (module_text, dims) in enumerate(modules):
    y_pos = hc_y - i * (module_height + 0.2)
    draw_box(ax_main, hc_x, y_pos, 3, module_height,
            module_text, color_handcrafted, fontsize=7)
    ax_main.text(hc_x + 3.2, y_pos + module_height/2, dims,
                fontsize=7, va='center', fontweight='bold')
    
    if i < len(modules) - 1:
        draw_arrow(ax_main, 2, y_pos, 2, y_pos - 0.2)

# Total features box
total_y = hc_y - len(modules) * (module_height + 0.2)
draw_box(ax_main, hc_x + 0.5, total_y, 2, 0.5,
        'Total: 511 Features', color_handcrafted, fontsize=8, fontweight='bold')

# Arrow to feature selection
selection_y = total_y - 0.8
draw_arrow(ax_main, 2, total_y, 2, selection_y + 0.5)
draw_box(ax_main, hc_x + 0.3, selection_y, 2.4, 0.5,
        'Feature Selection\n(Mutual Info)', '#FFE082', fontsize=7)

# Arrow to concatenation
concat_y = 1.5
draw_arrow(ax_main, 2, selection_y, 2, concat_y + 0.5)

# ============================================================================
# 4. RIGHT BRANCH - DEEP DNN ARCHITECTURE
# ============================================================================
dnn_x = 5.5
dnn_y = 6.5

ax_main.text(7.5, dnn_y + 0.7, 'Deep Residual DNN Classifier',
            ha='center', fontsize=10, fontweight='bold')

# Input layer
draw_box(ax_main, dnn_x, dnn_y, 2, 0.4,
        'Input: 360 Features\n(After Selection)', color_dnn, fontsize=8, fontweight='bold')

# Embedding layer
embed_y = dnn_y - 0.6
draw_arrow(ax_main, 6.5, dnn_y, 6.5, embed_y + 0.4)
draw_box(ax_main, dnn_x, embed_y, 2, 0.4,
        'Embedding Layer\nDense(512) → BatchNorm → Swish → Dropout(0.3)',
        color_dnn, fontsize=6)

# Residual blocks (simplified view)
residual_configs = [
    (512, 0.3, '1-2'),
    (384, 0.28, '3-4'),
    (256, 0.25, '5-6'),
    (128, 0.2, '7')
]

res_y = embed_y - 0.7
for i, (units, dropout, block_nums) in enumerate(residual_configs):
    y_pos = res_y - i * 0.65
    draw_box(ax_main, dnn_x + 0.2, y_pos, 1.6, 0.5,
            f'Residual Blocks {block_nums}\n{units} units, dropout={dropout}',
            color_dnn, fontsize=6.5)
    
    # Draw skip connection indicator
    ax_main.plot([dnn_x + 1.9, dnn_x + 2.1], [y_pos + 0.25, y_pos + 0.25],
                'r-', linewidth=2)
    ax_main.text(dnn_x + 2.15, y_pos + 0.25, 'Skip',
                fontsize=5, color='red', va='center')
    
    if i < len(residual_configs) - 1:
        draw_arrow(ax_main, 6.5, y_pos, 6.5, y_pos - 0.65 + 0.5)

# Attention layer
attention_y = res_y - len(residual_configs) * 0.65 - 0.3
draw_arrow(ax_main, 6.5, res_y - len(residual_configs) * 0.65 + 0.5, 6.5, attention_y + 0.5)
draw_box(ax_main, dnn_x, attention_y, 2, 0.5,
        'Self-Attention\nFeature Weighting (128-dim)',
        color_attention, fontsize=7, fontweight='bold')

# Final dense layer
final_y = attention_y - 0.6
draw_arrow(ax_main, 6.5, attention_y, 6.5, final_y + 0.4)
draw_box(ax_main, dnn_x + 0.3, final_y, 1.4, 0.4,
        'Dense(64)\nBatchNorm\nDropout(0.15)',
        color_dnn, fontsize=7)

# Arrow to concatenation
draw_arrow(ax_main, 6.5, final_y, 6.5, concat_y + 0.5)

# ============================================================================
# 5. FEATURE CONCATENATION (optional - or direct to output)
# ============================================================================
# Note: Based on your implementation, you can either:
# A) Train Deep DNN on handcrafted features (what you have)
# B) Concatenate CNN features with handcrafted (future enhancement)

# For current implementation (DNN trains on handcrafted features):
ax_main.text(5, concat_y + 0.9, 'Training Configuration',
            ha='center', fontsize=9, fontweight='bold')

draw_box(ax_main, 3.5, concat_y, 3, 0.4,
        '360 Selected Features → Deep DNN\nFocal Loss (α=0.16, γ=2.0) | Mixup (α=0.2)',
        '#FFF9C4', fontsize=7)

# ============================================================================
# 6. OUTPUT LAYER
# ============================================================================
output_y = 0.5
draw_arrow(ax_main, 5, concat_y, 5, output_y + 0.6)
draw_box(ax_main, 4, output_y, 2, 0.6,
        'Output Layer\nSigmoid Activation\nBCC vs SK',
        color_output, fontsize=9, fontweight='bold')

# Probability boxes
draw_box(ax_main, 2.5, output_y, 1.2, 0.3,
        'P(BCC)', '#C8E6C9', fontsize=8)
draw_box(ax_main, 6.3, output_y, 1.2, 0.3,
        'P(SK)', '#FFCCBC', fontsize=8)

# ============================================================================
# 7. DETAILED RESIDUAL BLOCK DIAGRAM (Inset)
# ============================================================================
ax_inset = fig.add_axes([0.72, 0.55, 0.22, 0.32])
ax_inset.set_xlim(0, 2)
ax_inset.set_ylim(0, 6)
ax_inset.axis('off')

ax_inset.text(1, 5.8, 'Residual Block Detail', ha='center', 
             fontsize=9, fontweight='bold')

draw_residual_block(ax_inset, 0.2, 0.2, 1.6, 5, 512, 0.3, 'i')

# ============================================================================
# 8. TRAINING HYPERPARAMETERS BOX
# ============================================================================
ax_hyper = fig.add_axes([0.02, 0.02, 0.25, 0.15])
ax_hyper.axis('off')

hyperparams_text = """Training Configuration:
• Optimizer: Adam (lr=1e-3)
• Loss: Focal Loss (α=0.16, γ=2.0)
• Regularization: L2 (1e-4), Dropout (0.2-0.3)
• Data Aug: Mixup (α=0.2)
• Batch Size: 128
• Early Stopping: Patience=30 (val_auc)
• Epochs: 200 (max)"""

ax_hyper.text(0.05, 0.95, hyperparams_text,
             transform=ax_hyper.transAxes,
             fontsize=7, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFFDE7', 
                      edgecolor='black', linewidth=1.5))

# ============================================================================
# 9. PERFORMANCE METRICS BOX
# ============================================================================
ax_metrics = fig.add_axes([0.75, 0.02, 0.23, 0.15])
ax_metrics.axis('off')

metrics_text = """Expected Performance:
• AUC: >99% (target)
• Accuracy: >98%
• Sensitivity: >97%
• Specificity: >98%
• F1-Score: >98%

Dataset: Balanced BCC/SK
Classes: BCC (Basal Cell Carcinoma)
         SK (Seborrheic Keratosis)"""

ax_metrics.text(0.05, 0.95, metrics_text,
               transform=ax_metrics.transAxes,
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#E8F5E9',
                        edgecolor='black', linewidth=1.5))

# ============================================================================
# 10. LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input/Preprocessing'),
    mpatches.Patch(facecolor=color_handcrafted, edgecolor='black', label='Handcrafted Features'),
    mpatches.Patch(facecolor=color_dnn, edgecolor='black', label='Deep DNN Layers'),
    mpatches.Patch(facecolor=color_attention, edgecolor='black', label='Attention Mechanism'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output/Prediction'),
    mpatches.Patch(facecolor='white', edgecolor='red', label='Skip Connection')
]

ax_main.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(0.02, 0.98), fontsize=7, framealpha=0.9)

# Save figure
output_path = 'output/figures/dnn_architecture_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Architecture diagram saved to: {output_path}")

output_path_pdf = 'output/figures/dnn_architecture_diagram.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ PDF version saved to: {output_path_pdf}")

plt.show()

print("\n" + "="*70)
print("Architecture figure generated successfully!")
print("="*70)
print("\nFigure shows:")
print("1. ✓ Input preprocessing pipeline")
print("2. ✓ Handcrafted feature extraction (left branch)")
print("3. ✓ Deep DNN with residual blocks (right branch)")
print("4. ✓ Self-attention mechanism")
print("5. ✓ Detailed residual block structure (inset)")
print("6. ✓ Training hyperparameters")
print("7. ✓ Expected performance metrics")
print("8. ✓ Skip connections visualization")
print("\nReady for your academic paper! 📄")
