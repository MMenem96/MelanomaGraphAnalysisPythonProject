import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
import os
import logging

class ModelEvaluator:
    def __init__(self, output_dir='output'):
        """Initialize the model evaluator with output directory for visualizations."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate_classifier(self, classifier, X, y, cv=5):
        """Comprehensive evaluation of a classifier with cross-validation."""
        try:
            self.logger.info("Starting comprehensive model evaluation...")
            
            # Ensure X and y are numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # 1. Perform k-fold cross-validation
            cv_results = self._perform_cross_validation(classifier, X, y, cv)
            
            # 2. Get predictions using the trained model
            # We'll use the best model from cross-validation if available, otherwise the input classifier
            if hasattr(classifier, 'best_estimator_'):
                best_model = classifier.best_estimator_
            else:
                best_model = classifier
                
            y_pred = best_model.predict(X)
            
            # 3. Get probability predictions (for ROC and PR curves)
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                self.logger.warning("Model doesn't support probability estimation, using decision function.")
                if hasattr(best_model, 'decision_function'):
                    y_proba = best_model.decision_function(X)
                else:
                    self.logger.warning("Model doesn't support decision function either. ROC/PR analysis will be limited.")
                    y_proba = y_pred.astype(float)
            
            # 4. Compute and display all metrics
            metrics = self._compute_metrics(y, y_pred, y_proba)
            
            # 5. Generate visualizations
            self._create_visualizations(y, y_pred, y_proba)
            
            # Return all results
            results = {
                'cv_results': cv_results,
                'metrics': metrics
            }
            
            self.logger.info("Model evaluation completed successfully.")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def _perform_cross_validation(self, classifier, X, y, cv=5):
        """Perform k-fold cross validation and return detailed results."""
        try:
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Initialize result arrays
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            specificities = []
            balanced_accuracies = []
            aucs = []
            auc_prs = []
            mccs = []
            kappas = []
            
            fold_metrics = []
            
            # Iterate through each fold
            for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model = classifier.__class__(**classifier.get_params())
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities for AUC calculation
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = y_pred.astype(float)
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)  # Same as sensitivity
                f1 = f1_score(y_test, y_pred)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_acc = (recall + specificity) / 2
                
                # AUC and AUC-PR
                auc = roc_auc_score(y_test, y_proba)
                auc_pr = average_precision_score(y_test, y_proba)
                
                # MCC and Kappa
                mcc = matthews_corrcoef(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                
                # Store metrics
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                specificities.append(specificity)
                balanced_accuracies.append(balanced_acc)
                aucs.append(auc)
                auc_prs.append(auc_pr)
                mccs.append(mcc)
                kappas.append(kappa)
                
                # Store all metrics for this fold
                fold_metrics.append({
                    'fold': i+1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'balanced_accuracy': balanced_acc,
                    'auc': auc,
                    'auc_pr': auc_pr,
                    'mcc': mcc,
                    'kappa': kappa,
                    'confusion_matrix': {
                        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                    }
                })
                
                self.logger.info(f"Fold {i+1} metrics: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
            
            # Calculate mean and std for each metric
            cv_summary = {
                'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
                'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
                'recall': {'mean': np.mean(recalls), 'std': np.std(recalls)},
                'f1_score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
                'specificity': {'mean': np.mean(specificities), 'std': np.std(specificities)},
                'balanced_accuracy': {'mean': np.mean(balanced_accuracies), 'std': np.std(balanced_accuracies)},
                'auc': {'mean': np.mean(aucs), 'std': np.std(aucs)},
                'auc_pr': {'mean': np.mean(auc_prs), 'std': np.std(auc_prs)},
                'mcc': {'mean': np.mean(mccs), 'std': np.std(mccs)},
                'kappa': {'mean': np.mean(kappas), 'std': np.std(kappas)}
            }
            
            # Log results
            self.logger.info(f"Cross-validation results ({cv}-fold):")
            for metric, values in cv_summary.items():
                self.logger.info(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")
            
            # Save results to file
            self._save_cv_results_to_file(cv_summary, fold_metrics)
            
            return {
                'summary': cv_summary,
                'fold_metrics': fold_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise
    
    def _compute_metrics(self, y_true, y_pred, y_proba):
        """Compute and return all evaluation metrics."""
        try:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)  # Same as sensitivity
            f1 = f1_score(y_true, y_pred)
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (recall + specificity) / 2
            
            # AUC
            auc = roc_auc_score(y_true, y_proba)
            
            # AUC-PR
            auc_pr = average_precision_score(y_true, y_proba)
            
            # MCC and Kappa
            mcc = matthews_corrcoef(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Return all metrics
            metrics = {
                'confusion_matrix': {
                    'matrix': cm.tolist(),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                },
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'balanced_accuracy': float(balanced_acc),
                'auc': float(auc),
                'auc_pr': float(auc_pr),
                'mcc': float(mcc),
                'kappa': float(kappa)
            }
            
            # Log results
            self.logger.info("Final evaluation metrics:")
            self.logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            self.logger.info(f"Accuracy: {accuracy:.3f}")
            self.logger.info(f"Precision: {precision:.3f}")
            self.logger.info(f"Recall (Sensitivity): {recall:.3f}")
            self.logger.info(f"Specificity: {specificity:.3f}")
            self.logger.info(f"F1 Score: {f1:.3f}")
            self.logger.info(f"Balanced Accuracy: {balanced_acc:.3f}")
            self.logger.info(f"AUC (ROC): {auc:.3f}")
            self.logger.info(f"AUC-PR: {auc_pr:.3f}")
            self.logger.info(f"Matthews Correlation Coefficient: {mcc:.3f}")
            self.logger.info(f"Cohen's Kappa: {kappa:.3f}")
            
            # Save metrics to file
            self._save_metrics_to_file(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def _create_visualizations(self, y_true, y_pred, y_proba):
        """Create and save visualization plots."""
        try:
            # Set style for plots
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Confusion Matrix Heatmap
            self._plot_confusion_matrix(y_true, y_pred)
            
            # 2. ROC Curve
            self._plot_roc_curve(y_true, y_proba)
            
            # 3. Precision-Recall Curve
            self._plot_pr_curve(y_true, y_proba)
            
            self.logger.info(f"Visualization plots saved to {self.output_dir} directory")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix as heatmap."""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate percentages for annotations
            cm_sum = np.sum(cm)
            cm_percentages = cm / cm_sum * 100
            
            # Get integer values for annotations
            tn, fp, fn, tp = cm.ravel()
            
            # Create annotation text
            annotations = [
                f'TN: {tn}\n({cm_percentages[0, 0]:.1f}%)',
                f'FP: {fp}\n({cm_percentages[0, 1]:.1f}%)',
                f'FN: {fn}\n({cm_percentages[1, 0]:.1f}%)',
                f'TP: {tp}\n({cm_percentages[1, 1]:.1f}%)'
            ]
            
            # Reshape to match the shape of confusion matrix
            annotations = np.array(annotations).reshape(2, 2)
            
            # Create heatmap
            ax = sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                           xticklabels=['Benign', 'Melanoma'],
                           yticklabels=['Benign', 'Melanoma'])
            
            plt.title('Confusion Matrix', fontsize=15)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def _plot_roc_curve(self, y_true, y_proba):
        """Plot and save ROC curve with AUC."""
        try:
            plt.figure(figsize=(8, 6))
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
            
            # Plot diagonal reference line
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
            
            # Mark some thresholds
            thresholds_to_mark = [0.3, 0.5, 0.7]
            for threshold in thresholds_to_mark:
                # Find closest threshold
                idx = np.abs(thresholds - threshold).argmin()
                # Plot marker
                plt.plot(fpr[idx], tpr[idx], 'ro')
                # Add annotation
                plt.annotate(f'Threshold: {thresholds[idx]:.2f}', 
                           (fpr[idx], tpr[idx]), 
                           xytext=(10, -10), 
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->'))
            
            # Set plot attributes
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=15)
            plt.legend(loc='lower right', fontsize=10)
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def _plot_pr_curve(self, y_true, y_proba):
        """Plot and save Precision-Recall curve with AUC-PR."""
        try:
            plt.figure(figsize=(8, 6))
            
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            
            # Plot PR curve
            plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {auc_pr:.3f})')
            
            # Plot baseline reference line (proportion of positive class)
            baseline = np.sum(y_true) / len(y_true)
            plt.plot([0, 1], [baseline, baseline], 'k--', lw=1, label=f'Baseline ({baseline:.3f})')
            
            # Mark some thresholds
            # We need to handle differently as thresholds for PR curve doesn't align with recall/precision
            thresholds_to_mark = [0.3, 0.5, 0.7]
            for threshold in thresholds_to_mark:
                if len(thresholds) > 0:
                    # Find closest threshold
                    idx = np.abs(thresholds - threshold).argmin()
                    if idx < len(precision) - 1:  # Ensure index is valid
                        # Plot marker
                        plt.plot(recall[idx], precision[idx], 'ro')
                        # Add annotation
                        plt.annotate(f'Threshold: {thresholds[idx]:.2f}', 
                                   (recall[idx], precision[idx]), 
                                   xytext=(10, 10), 
                                   textcoords='offset points',
                                   arrowprops=dict(arrowstyle='->'))
            
            # Set plot attributes
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall (Sensitivity)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=15)
            plt.legend(loc='best', fontsize=10)
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting PR curve: {str(e)}")
            raise
    
    def _save_metrics_to_file(self, metrics):
        """Save metrics to a text file."""
        try:
            output_file = os.path.join(self.output_dir, 'evaluation_metrics.txt')
            
            with open(output_file, 'w') as f:
                f.write("MELANOMA DETECTION MODEL EVALUATION\n")
                f.write("==================================\n\n")
                
                # Confusion Matrix
                cm = metrics['confusion_matrix']
                f.write("CONFUSION MATRIX:\n")
                f.write(f"True Negative (TN): {cm['tn']}\n")
                f.write(f"False Positive (FP): {cm['fp']}\n")
                f.write(f"False Negative (FN): {cm['fn']}\n")
                f.write(f"True Positive (TP): {cm['tp']}\n\n")
                
                # Core Metrics
                f.write("CLASSIFICATION METRICS:\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall (Sensitivity): {metrics['recall']:.4f}\n")
                f.write(f"Specificity: {metrics['specificity']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n\n")
                
                # Advanced Metrics
                f.write("ADVANCED METRICS:\n")
                f.write(f"AUC (ROC): {metrics['auc']:.4f}\n")
                f.write(f"AUC-PR: {metrics['auc_pr']:.4f}\n")
                f.write(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}\n")
                f.write(f"Cohen's Kappa: {metrics['kappa']:.4f}\n")
                
            self.logger.info(f"Metrics saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to file: {str(e)}")
            raise
    
    def _save_cv_results_to_file(self, cv_summary, fold_metrics):
        """Save cross-validation results to a text file."""
        try:
            output_file = os.path.join(self.output_dir, 'cross_validation_results.txt')
            
            with open(output_file, 'w') as f:
                f.write("CROSS-VALIDATION RESULTS\n")
                f.write("========================\n\n")
                
                # Summary statistics
                f.write("SUMMARY STATISTICS:\n")
                for metric, values in cv_summary.items():
                    f.write(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}\n")
                
                f.write("\n")
                
                # Per-fold results
                f.write("DETAILED FOLD RESULTS:\n")
                for fold in fold_metrics:
                    f.write(f"\nFold {fold['fold']}:\n")
                    f.write(f"  Accuracy: {fold['accuracy']:.4f}\n")
                    f.write(f"  Precision: {fold['precision']:.4f}\n")
                    f.write(f"  Recall: {fold['recall']:.4f}\n")
                    f.write(f"  F1 Score: {fold['f1_score']:.4f}\n")
                    f.write(f"  Specificity: {fold['specificity']:.4f}\n")
                    f.write(f"  Balanced Accuracy: {fold['balanced_accuracy']:.4f}\n")
                    f.write(f"  AUC: {fold['auc']:.4f}\n")
                    f.write(f"  AUC-PR: {fold['auc_pr']:.4f}\n")
                    f.write(f"  MCC: {fold['mcc']:.4f}\n")
                    f.write(f"  Kappa: {fold['kappa']:.4f}\n")
                    cm = fold['confusion_matrix']
                    f.write(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}\n")
                
            self.logger.info(f"Cross-validation results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving cross-validation results to file: {str(e)}")
            raise