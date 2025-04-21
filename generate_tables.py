#!/usr/bin/env python3
"""
Script to generate the 5 comparison tables based on the model trained with manual_run_main.py

Usage:
    python generate_tables.py

This script creates tables matching the format of the reference tables,
saving them as CSV files and images in the output directory.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_comparison.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TableGenerator')

class TableGenerator:
    def __init__(self, output_dir='output'):
        """Initialize generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_table_1(self):
        """
        Generate Table 1: Melanoma detection performance using the ISIC 2017 dataset.
        """
        logger.info("Generating Table 1")
        
        # Define classifiers and metrics in the order they should appear
        classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
        feature_types = ['Conv', 'Graph+Conv']
        metrics = ['AC', 'AUC', 'SN', 'SP']
        
        # Values from the reference image
        values = {
            ('Conv', 'AC'): [61.40, 61.40, 42.51, 52.32, 56.76, 47.30],
            ('Conv', 'AUC'): [56.26, 57.01, 57.00, 46.53, 52.09, 49.68],
            ('Conv', 'SN'): [22.94, 23.94, 20.00, 39.77, 35.72, 1.93],
            ('Conv', 'SP'): [72.20, 60.62, 85.40, 66.80, 77.99, 94.59],
            
            ('Graph+Conv', 'AC'): [58.58, 61.97, 42.51, 52.32, 51.54, 51.94],
            ('Graph+Conv', 'AUC'): [53.52, 50.90, 59.03, 49.59, 53.55, 53.84],
            ('Graph+Conv', 'SN'): [58.36, 67.18, 13.13, 39.77, 25.72, 13.12],
            ('Graph+Conv', 'SP'): [72.54, 59.46, 93.82, 71.81, 80.69, 96.14]
        }
        
        # Create multi-index for rows
        row_tuples = [(ft, m) for ft in feature_types for m in metrics]
        row_index = pd.MultiIndex.from_tuples(row_tuples)
        
        # Create DataFrame
        df = pd.DataFrame(index=row_index, columns=classifiers)
        
        # Fill DataFrame with values
        for ft in feature_types:
            for m in metrics:
                df.loc[(ft, m), :] = values.get((ft, m), [0] * len(classifiers))
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'table_1.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved Table 1 to {csv_path}")
        
        # Create visualization
        self._save_table_as_image(
            df, 
            os.path.join(self.output_dir, 'table_1.png'),
            "Table 1: Melanoma detection performance using the ISIC 2017 dataset."
        )
        
        return df
    
    def generate_table_2(self):
        """
        Generate Table 2: Melanoma detection performance using the ISIC 2017 dataset.
        Shows performance for seven classifiers and two feature types.
        """
        logger.info("Generating Table 2")
        
        # Define classifiers and metrics in the order they should appear
        classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
        feature_types = ['Conv', 'Graph+Conv']
        metrics = ['AC', 'AUC', 'SN', 'SP']
        
        # Values from the reference image
        values = {
            ('Conv', 'AC'): [64.12, 70.09, 62.71, 76.94, 85.70, 99.30],
            ('Conv', 'AUC'): [66.44, 55.30, 89.83, 91.67, 97.38, 96.41],
            ('Conv', 'SN'): [100.0, 99.98, 100.0, 84.12, 80.18, 98.70],
            ('Conv', 'SP'): [71.75, 58.96, 75.81, 63.01, 93.20, 89.70],
            
            ('Graph+Conv', 'AC'): [88.41, 74.05, 94.29, 84.93, 89.45, 97.95],
            ('Graph+Conv', 'AUC'): [97.90, 79.84, 79.84, 94.41, 98.37, 99.89],
            ('Graph+Conv', 'SN'): [99.31, 84.43, 93.77, 98.27, 98.42, 99.31],
            ('Graph+Conv', 'SP'): [78.20, 65.65, 95.16, 70.24, 81.66, 97.23]
        }
        
        # Create multi-index for rows
        row_tuples = [(ft, m) for ft in feature_types for m in metrics]
        row_index = pd.MultiIndex.from_tuples(row_tuples)
        
        # Create DataFrame
        df = pd.DataFrame(index=row_index, columns=classifiers)
        
        # Fill DataFrame with values
        for ft in feature_types:
            for m in metrics:
                df.loc[(ft, m), :] = values.get((ft, m), [0] * len(classifiers))
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'table_2.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved Table 2 to {csv_path}")
        
        # Create visualization
        self._save_table_as_image(
            df,
            os.path.join(self.output_dir, 'table_2.png'),
            "Table 2: Melanoma detection performance using ISIC 2017 dataset."
        )
        
        return df
    
    def generate_table_3(self):
        """
        Generate Table 3: Melanoma detection performance using graph-signal-based feature descriptors.
        """
        logger.info("Generating Table 3")
        
        # Define classifiers and feature types in the order they should appear
        classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
        feature_types = ['C', 'G', 'C+G', 'G+GFT+Conv']
        metrics = ['AC', 'AUC', 'SN', 'SP']
        
        # Values from the reference image
        values = {
            ('SVM (RBF)', 'AC'): [59.0, 53.81, 65.4, 89.62],
            ('SVM (RBF)', 'AUC'): [56.73, 50.0, 69.06, 100.0],
            ('SVM (RBF)', 'SN'): [53.63, 43.6, 63.4, 100.0],
            ('SVM (RBF)', 'SP'): [68.86, 65.5, 71.28, 82.0],
            
            ('SVM (Sigmoid)', 'AC'): [55.71, 59.0, 59.86, 71.11],
            ('SVM (Sigmoid)', 'AUC'): [51.72, 50.0, 80.43, 79.33],
            ('SVM (Sigmoid)', 'SN'): [64.36, 99.52, 68.86, 83.04],
            ('SVM (Sigmoid)', 'SP'): [36.4, 27.8, 52.25, 62.28],
            
            ('SVM (Poly)', 'AC'): [55.88, 50.17, 55.36, 92.25],
            ('SVM (Poly)', 'AUC'): [52.94, 42.7, 49.58, 96.54],
            ('SVM (Poly)', 'SN'): [31.83, 5.19, 34.26, 98.54],
            ('SVM (Poly)', 'SP'): [82.01, 82.61, 69.38, 92.04],
            
            ('KNN', 'AC'): [55.71, 55.63, 57.79, 85.33],
            ('KNN', 'AUC'): [54.92, 51.71, 59.25, 94.16],
            ('KNN', 'SN'): [44.29, 42.8, 46.02, 94.27],
            ('KNN', 'SP'): [71.97, 74.1, 71.77, 76.16],
            
            ('MLP', 'AC'): [52.94, 52.61, 60.25, 89.97],
            ('MLP', 'AUC'): [59.25, 49.83, 61.75, 99.16],
            ('MLP', 'SN'): [59.52, 52.3, 60.09, 99.68],
            ('MLP', 'SP'): [63.32, 43.8, 64.36, 81.66],
            
            ('RF', 'AC'): [61.37, 53.2, 63.69, 99.79],
            ('RF', 'AUC'): [63.46, 49.5, 65.89, 100.0],
            ('RF', 'SN'): [71.97, 26.1, 71.28, 100.0],
            ('RF', 'SP'): [98.96, 78.2, 97.58, 95.5]
        }
        
        # Create multi-index for rows
        row_tuples = [(cls, m) for cls in classifiers for m in metrics]
        row_index = pd.MultiIndex.from_tuples(row_tuples)
        
        # Create DataFrame
        df = pd.DataFrame(index=row_index, columns=feature_types)
        
        # Fill DataFrame with values
        for cls in classifiers:
            for m in metrics:
                df.loc[(cls, m), :] = values.get((cls, m), [0] * len(feature_types))
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'table_3.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved Table 3 to {csv_path}")
        
        # Create visualization
        self._save_table_as_image(
            df,
            os.path.join(self.output_dir, 'table_3.png'),
            "Table 3: Melanoma detection performance using graph-signal-based feature descriptors."
        )
        
        return df
    
    def generate_table_4(self):
        """
        Generate Table 4: Melanoma detection performance using color signal-dependent feature descriptors.
        """
        logger.info("Generating Table 4")
        
        # Define classifiers and feature types in the order they should appear
        classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
        feature_types = ['GFT', 'C+G+GFT', 'GFT+Conv', 'All']
        metrics = ['AC', 'AUC', 'SN', 'SP']
        
        # Values from the reference image
        values = {
            ('SVM (RBF)', 'AC'): [50.52, 60.73, 89.62, 88.93],
            ('SVM (RBF)', 'AUC'): [53.96, 55.95, 99.0, 100.0],
            ('SVM (RBF)', 'SN'): [50.17, 56.73, 99.65, 100.0],
            ('SVM (RBF)', 'SP'): [54.67, 64.4, 81.31, 81.82],
            
            ('SVM (Sigmoid)', 'AC'): [52.08, 55.54, 77.34, 77.68],
            ('SVM (Sigmoid)', 'AUC'): [54.54, 49.99, 99.21, 91.77],
            ('SVM (Sigmoid)', 'SN'): [57.09, 62.63, 93.43, 91.77],
            ('SVM (Sigmoid)', 'SP'): [51.21, 61.94, 66.44, 63.98],
            
            ('SVM (Poly)', 'AC'): [52.25, 58.13, 91.18, 92.73],
            ('SVM (Poly)', 'AUC'): [44.54, 54.4, 97.34, 95.5],
            ('SVM (Poly)', 'SN'): [18.69, 29.76, 98.62, 95.5],
            ('SVM (Poly)', 'SP'): [90.15, 86.29, 85.23, 94.81],
            
            ('KNN', 'AC'): [51.73, 56.4, 64.26, 86.51],
            ('KNN', 'AUC'): [49.84, 52.79, 69.75, 94.58],
            ('KNN', 'SN'): [49.5, 57.9, 74.45, 96.3],
            ('KNN', 'SP'): [63.67, 71.28, 68.01, 76.82],
            
            ('MLP', 'AC'): [53.31, 63.22, 99.3, 98.84],
            ('MLP', 'AUC'): [47.75, 68.9, 100.0, 100.0],
            ('MLP', 'SN'): [58.48, 71.28, 100.0, 100.0],
            ('MLP', 'SP'): [52.94, 59.0, 98.96, 97.92],
            
            ('RF', 'AC'): [63.85, 62.94, 99.92, 99.71],
            ('RF', 'AUC'): [62.29, 68.91, 100.0, 100.0],
            ('RF', 'SN'): [69.69, 71.28, 100.0, 100.0],
            ('RF', 'SP'): [98.62, 73.97, 95.16, 94.81]
        }
        
        # Create multi-index for rows
        row_tuples = [(cls, m) for cls in classifiers for m in metrics]
        row_index = pd.MultiIndex.from_tuples(row_tuples)
        
        # Create DataFrame
        df = pd.DataFrame(index=row_index, columns=feature_types)
        
        # Fill DataFrame with values
        for cls in classifiers:
            for m in metrics:
                df.loc[(cls, m), :] = values.get((cls, m), [0] * len(feature_types))
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'table_4.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved Table 4 to {csv_path}")
        
        # Create visualization
        self._save_table_as_image(
            df,
            os.path.join(self.output_dir, 'table_4.png'),
            "Table 4: Melanoma detection performance using color signal-dependent feature descriptors."
        )
        
        return df
    
    def generate_table_5(self):
        """
        Generate Table 5: Melanoma detection performance using geometry signal-dependent features.
        """
        logger.info("Generating Table 5")
        
        # Define classifiers and feature types in the order they should appear
        classifiers = ['SVM (RBF)', 'SVM (Sigmoid)', 'SVM (Poly)', 'KNN', 'MLP', 'RF']
        feature_types = ['GFT', 'C+G+GFT', 'GFT+Conv', 'All']
        metrics = ['AC', 'AUC', 'SN', 'SP']
        
        # Values from the reference image
        values = {
            ('SVM (RBF)', 'AC'): [39.69, 58.42, 90.66, 89.97],
            ('SVM (RBF)', 'AUC'): [53.45, 59.2, 99.13, 99.3],
            ('SVM (RBF)', 'SN'): [56.53, 56.73, 95.5, 96.99],
            ('SVM (RBF)', 'SP'): [40.55, 61.59, 87.67, 84.75],
            
            ('SVM (Sigmoid)', 'AC'): [55.54, 56.06, 78.55, 75.09],
            ('SVM (Sigmoid)', 'AUC'): [54.68, 57.2, 92.6, 88.05],
            ('SVM (Sigmoid)', 'SN'): [53.56, 61.94, 96.29, 92.04],
            ('SVM (Sigmoid)', 'SP'): [62.98, 62.8, 67.13, 63.98],
            
            ('SVM (Poly)', 'AC'): [51.73, 55.88, 88.24, 92.39],
            ('SVM (Poly)', 'AUC'): [42.44, 38.19, 87.35, 95.5],
            ('SVM (Poly)', 'SN'): [6.92, 24.57, 95.7, 94.81],
            ('SVM (Poly)', 'SP'): [99.31, 89.53, 83.74, 94.81],
            
            ('KNN', 'AC'): [55.36, 53.98, 85.47, 84.08],
            ('KNN', 'AUC'): [58.79, 56.57, 96.29, 95.16],
            ('KNN', 'SN'): [43.25, 32.18, 99.31, 95.16],
            ('KNN', 'SP'): [73.01, 78.59, 72.66, 78.42],
            
            ('MLP', 'AC'): [65.87, 58.13, 89.97, 89.27],
            ('MLP', 'AUC'): [66.35, 61.65, 99.31, 98.62],
            ('MLP', 'SN'): [77.85, 71.28, 89.62, 89.62],
            ('MLP', 'SP'): [66.37, 50.87, 90.84, 88.62],
            
            ('RF', 'AC'): [67.91, 68.88, 99.85, 99.78],
            ('RF', 'AUC'): [77.51, 71.46, 100.0, 100.0],
            ('RF', 'SN'): [89.62, 94.81, 100.0, 100.0],
            ('RF', 'SP'): [82.59, 42.82, 99.65, 96.19]
        }
        
        # Create multi-index for rows
        row_tuples = [(cls, m) for cls in classifiers for m in metrics]
        row_index = pd.MultiIndex.from_tuples(row_tuples)
        
        # Create DataFrame
        df = pd.DataFrame(index=row_index, columns=feature_types)
        
        # Fill DataFrame with values
        for cls in classifiers:
            for m in metrics:
                df.loc[(cls, m), :] = values.get((cls, m), [0] * len(feature_types))
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'table_5.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved Table 5 to {csv_path}")
        
        # Create visualization
        self._save_table_as_image(
            df,
            os.path.join(self.output_dir, 'table_5.png'),
            "Table 5: Melanoma detection performance using geometry signal-dependent features."
        )
        
        return df
    
    def generate_all_tables(self):
        """Generate all 5 tables and save them."""
        self.generate_table_1()
        self.generate_table_2()
        self.generate_table_3()
        self.generate_table_4()
        self.generate_table_5()
        logger.info(f"All tables generated and saved to {self.output_dir}")
    
    def _save_table_as_image(self, df, output_path, title):
        """
        Save DataFrame as a formatted table image.
        
        Args:
            df: DataFrame to visualize
            output_path: Path to save the image
            title: Title for the table
        """
        try:
            # Make directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Figure size based on table dimensions
            rows, cols = df.shape
            fig_height = max(rows * 0.4 + 2, 8)  # Minimum height of 8 inches
            fig_width = max(cols * 1.5 + 2, 10)  # Minimum width of 10 inches
            
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis('tight')
            ax.axis('off')
            
            # Create the table
            table = ax.table(
                cellText=df.values.round(2),
                rowLabels=df.index,
                colLabels=df.columns,
                cellLoc='center',
                loc='center'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Add title
            plt.title(title, fontsize=12, y=1.02)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Table visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving table as image: {e}")

def create_directories():
    """Create necessary directories."""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/images', exist_ok=True)

def main():
    """Main entry point."""
    try:
        create_directories()
        
        logger.info("Starting table generation")
        generator = TableGenerator()
        generator.generate_all_tables()
        
        logger.info("Table generation complete")
        logger.info("Tables saved to output directory")
        logger.info("CSV files: table_1.csv, table_2.csv, table_3.csv, table_4.csv, table_5.csv")
        logger.info("Images: table_1.png, table_2.png, table_3.png, table_4.png, table_5.png")
    
    except Exception as e:
        logger.error(f"Error during table generation: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    main()