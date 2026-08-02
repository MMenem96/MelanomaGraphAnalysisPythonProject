"""
Visualization of Machine Learning Classifiers Used in the Project
Creates a professional diagram showing all classifiers with consistent styling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path
import numpy as np


def visualize_ml_classifiers(save_path=None, dpi=300):
    """
    Create a professional visualization of all ML classifiers used in the project.
    Groups SVM variants together and uses colors consistent with DNN visualization.
    
    Args:
        save_path: Path to save the figure (default: output/architecture/ml_classifiers.png)
        dpi: Resolution for saved figure (default: 300)
    """
    
    # Define classifiers (SVM variants grouped together)
    classifiers = {
        'Tree-Based': [
            'Random Forest',
            'Extra Trees',
            'Gradient Boosting'
        ],
        'Boosting': [
            'XGBoost',
            'LightGBM',
            'CatBoost'
        ],
        'SVM': [
            'RBF Kernel',
            'Linear Kernel',
            'Polynomial Kernel',
            'Sigmoid Kernel'
        ]
    }
    
    # Define colors similar to DNN visualization (professional palette)
    category_colors = {
        'Tree-Based': '#CCE5FF',      # Light Blue
        'Boosting': '#FFFFCC',         # Light Yellow
        'SVM': '#FFCCFF',              # Light Pink
        'Neural Network': '#CCFFDD',   # Light Green
        'Others': '#FFE5CC'            # Peach
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Machine Learning Classifiers', 
           ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 9.1, 'Classification Models for Skin Lesion Analysis (BCC/BKL)', 
           ha='center', fontsize=12, color='#444')
    
    # Calculate positions for categories
    n_categories = len(classifiers)
    category_width = 1.6
    category_spacing = 0.3
    total_width = n_categories * category_width + (n_categories - 1) * category_spacing
    start_x = (10 - total_width) / 2
    
    y_start = 7.5
    
    # Draw each category
    for idx, (category, models) in enumerate(classifiers.items()):
        x_pos = start_x + idx * (category_width + category_spacing)
        
        # Calculate height based on number of models
        n_models = len(models)
        model_height = 0.5
        model_spacing = 0.15
        total_height = n_models * model_height + (n_models - 1) * model_spacing + 0.8
        
        y_bottom = y_start - total_height
        
        # Draw category box
        category_box = FancyBboxPatch(
            (x_pos, y_bottom), category_width, total_height,
            boxstyle="round,pad=0.08",
            facecolor=category_colors[category],
            edgecolor='#333',
            linewidth=2.0,
            alpha=0.7,
            zorder=1
        )
        ax.add_patch(category_box)
        
        # Draw category label
        ax.text(x_pos + category_width/2, y_start - 0.35, category,
               ha='center', va='center', fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='#333', linewidth=1.5))
        
        # Draw individual models
        model_y = y_start - 0.9
        for model in models:
            # Model box
            model_box = Rectangle(
                (x_pos + 0.15, model_y - model_height/2), 
                category_width - 0.3, model_height,
                facecolor='white',
                edgecolor='black',
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(model_box)
            
            # Model text
            ax.text(x_pos + category_width/2, model_y, model,
                   ha='center', va='center', fontsize=10, 
                   fontweight='normal')
            
            model_y -= (model_height + model_spacing)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        output_dir = Path('output/architecture')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'ml_classifiers.png'
    else:
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f"✅ ML classifiers diagram saved to: {save_path}")
    
    plt.close(fig)
    return fig


def visualize_evaluation_metrics(save_path=None, dpi=300):
    """
    Create a professional visualization of evaluation metrics used in the project.
    Shows all performance metrics with formulas and consistent styling.
    
    Args:
        save_path: Path to save the figure (default: output/architecture/evaluation_metrics.png)
        dpi: Resolution for saved figure (default: 300)
    """
    
    # Define metrics with their formulas
    metrics = {
        'Accuracy': {
            'formula': r'$\frac{TP + TN}{TP + TN + FP + FN}$',
            'description': 'Overall correctness',
            'color': '#CCE5FF'  # Light Blue
        },
        'Sensitivity': {
            'formula': r'$\frac{TP}{TP + FN}$',
            'description': 'True Positive Rate (Recall)',
            'color': '#FFFFCC'  # Light Yellow
        },
        'Specificity': {
            'formula': r'$\frac{TN}{TN + FP}$',
            'description': 'True Negative Rate',
            'color': '#CCFFDD'  # Light Green
        },
        'Precision': {
            'formula': r'$\frac{TP}{TP + FP}$',
            'description': 'Positive Predictive Value',
            'color': '#FFCCFF'  # Light Pink
        },
        'F1-Score': {
            'formula': r'$2 \times \frac{Precision \times Recall}{Precision + Recall}$',
            'description': 'Harmonic mean of Precision & Recall',
            'color': '#FFE5CC'  # Peach
        },
        'AUC': {
            'formula': r'$\int_{0}^{1} TPR(FPR^{-1}(x)) dx$',
            'description': 'Area Under ROC Curve',
            'color': '#FFE5DD'  # Light Peach
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Evaluation Metrics', 
           ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 9.1, 'Performance Measures for Binary Classification (BCC/BKL)', 
           ha='center', fontsize=12, color='#444')
    
    # Calculate positions for metrics (2 rows, 3 columns)
    n_cols = 3
    n_rows = 2
    metric_width = 2.8
    metric_height = 2.5
    h_spacing = 0.4
    v_spacing = 0.5
    
    total_width = n_cols * metric_width + (n_cols - 1) * h_spacing
    total_height = n_rows * metric_height + (n_rows - 1) * v_spacing
    
    start_x = (10 - total_width) / 2
    start_y = 8.0
    
    # Draw each metric
    metric_items = list(metrics.items())
    for idx, (metric_name, metric_info) in enumerate(metric_items):
        row = idx // n_cols
        col = idx % n_cols
        
        x_pos = start_x + col * (metric_width + h_spacing)
        y_pos = start_y - row * (metric_height + v_spacing)
        
        # Draw metric box
        metric_box = FancyBboxPatch(
            (x_pos, y_pos - metric_height), metric_width, metric_height,
            boxstyle="round,pad=0.1",
            facecolor=metric_info['color'],
            edgecolor='#333',
            linewidth=2.0,
            alpha=0.7,
            zorder=1
        )
        ax.add_patch(metric_box)
        
        # Metric name
        ax.text(x_pos + metric_width/2, y_pos - 0.4, metric_name,
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='#333', linewidth=1.5))
        
        # Formula
        ax.text(x_pos + metric_width/2, y_pos - 1.2, metric_info['formula'],
               ha='center', va='center', fontsize=13)
        
        # Description
        ax.text(x_pos + metric_width/2, y_pos - 2.0, metric_info['description'],
               ha='center', va='center', fontsize=10, style='italic', 
               color='#555', wrap=True)
    
    # Add confusion matrix reference at bottom
    cm_y = 1.2
    
    # Confusion matrix title
    ax.text(5, cm_y + 0.5, 'Confusion Matrix Components:', 
           ha='center', fontsize=12, fontweight='bold')
    
    # Draw small confusion matrix
    cm_size = 1.0
    cm_x = 3.5
    
    # Create 2x2 grid
    for i in range(2):
        for j in range(2):
            rect = Rectangle(
                (cm_x + j * cm_size, cm_y - 0.5 - i * cm_size),
                cm_size, cm_size,
                facecolor='white' if (i + j) % 2 == 0 else '#f0f0f0',
                edgecolor='black',
                linewidth=2.0
            )
            ax.add_patch(rect)
    
    # Labels
    ax.text(cm_x + 0.5, cm_y - 0.5 - 0, 'TP', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(cm_x + 1.5, cm_y - 0.5 - 0, 'FP', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(cm_x + 0.5, cm_y - 0.5 - 1.0, 'FN', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(cm_x + 1.5, cm_y - 0.5 - 1.0, 'TN', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Axis labels
    ax.text(cm_x - 0.3, cm_y - 0.5, 'Predicted', ha='right', va='center', 
           fontsize=10, fontweight='bold', rotation=90)
    ax.text(cm_x + 1.0, cm_y + 0.1, 'Actual', ha='center', va='bottom', 
           fontsize=10, fontweight='bold')
    
    # Legend
    legend_x = 6.5
    legend_items = [
        'TP: True Positive',
        'TN: True Negative',
        'FP: False Positive',
        'FN: False Negative'
    ]
    
    for i, item in enumerate(legend_items):
        ax.text(legend_x, cm_y + 0.2 - i * 0.35, f'• {item}', 
               ha='left', va='center', fontsize=10, color='#333')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        output_dir = Path('output/architecture')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'evaluation_metrics.png'
    else:
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f"✅ Evaluation metrics diagram saved to: {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == '__main__':
    visualize_ml_classifiers()
    visualize_evaluation_metrics()
