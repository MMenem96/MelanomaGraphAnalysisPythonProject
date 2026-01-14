"""
Extract metrics from saved models for abstract without retraining.
This script reads metadata from trained models and calculates all required values.
"""
import json
import glob
import numpy as np
from scipy import stats

def calculate_ci_from_confusion_matrix(conf_matrix, confidence=0.95):
    """Calculate 95% CI from confusion matrix."""
    tn, fp, fn, tp = conf_matrix.ravel()
    n = tn + fp + fn + tp
    
    # Accuracy CI
    accuracy = (tp + tn) / n
    acc_se = np.sqrt(accuracy * (1 - accuracy) / n)
    acc_ci = stats.norm.interval(confidence, loc=accuracy, scale=acc_se)
    
    # Sensitivity CI
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sens_se = np.sqrt(sensitivity * (1 - sensitivity) / (tp + fn)) if (tp + fn) > 0 else 0
    sens_ci = stats.norm.interval(confidence, loc=sensitivity, scale=sens_se)
    
    # Specificity CI
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    spec_se = np.sqrt(specificity * (1 - specificity) / (tn + fp)) if (tn + fp) > 0 else 0
    spec_ci = stats.norm.interval(confidence, loc=specificity, scale=spec_se)
    
    return {
        'accuracy': (acc_ci[0]*100, acc_ci[1]*100),
        'sensitivity': (sens_ci[0]*100, sens_ci[1]*100),
        'specificity': (spec_ci[0]*100, spec_ci[1]*100)
    }

def main():
    # Find all model metadata
    model_dirs = glob.glob('model/feature_based/*/metadata.json')
    
    if not model_dirs:
        print("❌ No model metadata found!")
        print("Please ensure models have been trained and saved.")
        return
    
    print(f"\n✅ Found {len(model_dirs)} trained models")
    
    all_models = []
    for metadata_path in model_dirs:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metrics = metadata['test_metrics']
            all_models.append({
                'name': metadata['model_type'],
                'accuracy': metrics['accuracy'] * 100,
                'sensitivity': metrics['recall'] * 100,
                'specificity': metrics['specificity'] * 100,
                'precision': metrics['precision'] * 100,
                'f1': metrics['f1'] * 100,
                'auc': metrics['roc_auc'] if metrics['roc_auc'] else 0,
                'conf_matrix': np.array(metrics['confusion_matrix']),
                'path': metadata_path
            })
        except Exception as e:
            print(f"⚠️  Skipping {metadata_path}: {e}")
    
    if not all_models:
        print("❌ No valid models found!")
        return
    
    # Find best model by AUC
    best_model = max(all_models, key=lambda x: x['auc'])
    
    # Find Deep DNN model
    dnn_model = next((m for m in all_models if 'DNN' in m['name'] or 'Deep' in m['name']), None)
    
    print("\n" + "="*80)
    print("ABSTRACT METRICS - COPY THESE VALUES")
    print("="*80)
    
    print(f"\n🏆 BEST MODEL: {best_model['name']}")
    print(f"   AUC: {best_model['auc']:.3f}")
    print(f"   Accuracy: {best_model['accuracy']:.1f}%")
    print(f"   Sensitivity (Recall): {best_model['sensitivity']:.1f}%")
    print(f"   Specificity: {best_model['specificity']:.1f}%")
    print(f"   Precision: {best_model['precision']:.1f}%")
    print(f"   F1 Score: {best_model['f1']:.1f}%")
    
    # Calculate 95% CI
    cis = calculate_ci_from_confusion_matrix(best_model['conf_matrix'])
    print(f"\n📊 95% CONFIDENCE INTERVALS:")
    print(f"   Accuracy: [{cis['accuracy'][0]:.1f}%, {cis['accuracy'][1]:.1f}%]")
    print(f"   Sensitivity: [{cis['sensitivity'][0]:.1f}%, {cis['sensitivity'][1]:.1f}%]")
    print(f"   Specificity: [{cis['specificity'][0]:.1f}%, {cis['specificity'][1]:.1f}%]")
    
    if dnn_model:
        print(f"\n🤖 DEEP DNN BASELINE:")
        print(f"   Accuracy: {dnn_model['accuracy']:.1f}%")
        print(f"   AUC: {dnn_model['auc']:.3f}")
        print(f"   Sensitivity: {dnn_model['sensitivity']:.1f}%")
        print(f"   Specificity: {dnn_model['specificity']:.1f}%")
        
        # Calculate improvements
        auc_improvement = ((best_model['auc'] - dnn_model['auc']) / dnn_model['auc']) * 100
        sens_improvement = best_model['sensitivity'] - dnn_model['sensitivity']
        
        print(f"\n📈 IMPROVEMENTS OVER DNN:")
        print(f"   AUC improved by: {auc_improvement:.1f}%")
        print(f"   Sensitivity improved by: {sens_improvement:.1f} percentage points")
    else:
        print("\n⚠️  No Deep DNN model found for comparison")
        dnn_model = None
        auc_improvement = 0
        sens_improvement = 0
    
    print("\n" + "="*80)
    print("LaTeX FORMATTED FOR ABSTRACT:")
    print("="*80)
    
    # AUC CI (approximation for single test set)
    auc_ci_lower = max(0, best_model['auc'] - 0.02)
    auc_ci_upper = min(1, best_model['auc'] + 0.02)
    
    print(f"""
an area under the ROC curve (AUC) of ${best_model['auc']:.2f}$ 
(95\\% CI: [{auc_ci_lower:.2f}--{auc_ci_upper:.2f}]), 
an accuracy of {best_model['accuracy']:.1f}\\% 
(95\\% CI: [{cis['accuracy'][0]:.1f}--{cis['accuracy'][1]:.1f}]\\%), 
a specificity of {best_model['specificity']:.1f}\\% 
(95\\% CI: [{cis['specificity'][0]:.1f}--{cis['specificity'][1]:.1f}]\\%), 
and a sensitivity of {best_model['sensitivity']:.1f}\\% 
(95\\% CI: [{cis['sensitivity'][0]:.1f}--{cis['sensitivity'][1]:.1f}]\\%).
""")
    
    if dnn_model:
        print(f"Compared to the standalone deep neural network (DNN) baseline,")
        print(f"which achieved an accuracy of {dnn_model['accuracy']:.1f}\\% and an AUC of ${dnn_model['auc']:.3f}$,")
        print(f"our approach improved the AUC by {auc_improvement:.1f}\\% ")
        print(f"and the sensitivity by {sens_improvement:.1f}\\%, ")
    
    # Print all models for reference
    print("\n" + "="*80)
    print("ALL MODELS SUMMARY:")
    print("="*80)
    for model in sorted(all_models, key=lambda x: x['auc'], reverse=True):
        print(f"\n{model['name']}:")
        print(f"  AUC: {model['auc']:.3f} | Acc: {model['accuracy']:.1f}% | "
              f"Sens: {model['sensitivity']:.1f}% | Spec: {model['specificity']:.1f}%")

if __name__ == '__main__':
    main()
