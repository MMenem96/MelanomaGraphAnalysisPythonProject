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
            # Count samples in each class to determine appropriate CV
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Adjust CV to be at most the minimum class count
            effective_cv = min(cv, min_class_count)
            
            if effective_cv < cv:
                self.logger.warning(f"Reducing CV folds from {cv} to {effective_cv} due to limited samples in smallest class")
                
            # If we have very few samples, use Leave-One-Out CV (effective_cv=1)
            if effective_cv < 2:
                effective_cv = 2  # Minimum of 2 folds
                self.logger.warning(f"Using minimum 2-fold CV due to very limited samples")
            
            kfold = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=42)
            
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
                'kappa': {'mean': np.mean(kappas), 'std': np.std(kappas)},
                'fold_metrics': fold_metrics
            }
            
            return cv_summary
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def _compute_metrics(self, y_true, y_pred, y_proba):
        """Compute comprehensive set of evaluation metrics."""
        try:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)  # Same as sensitivity
            f1 = f1_score(y_true, y_pred)
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Balanced accuracy
            balanced_acc = (recall + specificity) / 2
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            
            # Additional metrics
            mcc = matthews_corrcoef(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Collect all metrics
            metrics = {
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
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'matrix': cm.tolist()
                },
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                },
                'pr_curve': {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist()
                }
            }
            
            # Log primary metrics
            self.logger.info(f"Accuracy: {accuracy:.3f}")
            self.logger.info(f"Precision: {precision:.3f}")
            self.logger.info(f"Recall: {recall:.3f}")
            self.logger.info(f"F1 Score: {f1:.3f}")
            self.logger.info(f"AUC: {auc:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def _create_visualizations(self, y_true, y_pred, y_proba):
        """Create and save evaluation visualizations."""
        try:
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            self._plot_confusion_matrix(cm, ['Benign', 'Melanoma'])
            
            # 2. ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            self._plot_roc_curve(fpr, tpr, auc)
            
            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
            self._plot_precision_recall_curve(precision, recall, auc_pr)
            
            # 4. Feature importance (if available)
            self._plot_feature_importance_if_available(y_true, y_pred)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix visualization."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    def _plot_roc_curve(self, fpr, tpr, auc):
        """Plot and save ROC curve visualization."""
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
    
    def _plot_precision_recall_curve(self, precision, recall, auc_pr):
        """Plot and save precision-recall curve visualization."""
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='green', lw=2, 
                    label=f'Precision-Recall curve (AP = {auc_pr:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curve: {str(e)}")
    
    def _plot_feature_importance_if_available(self, y_true, y_pred):
        """
        Plot feature importance if classifier supports it.
        This method is a placeholder and should be implemented based on the specific 
        classifier being used.
        """
        # This would be implemented when we have a classifier that provides feature importance
        pass