"""
Quick test script for TabularDNNClassifier
Tests the classifier on synthetic data to verify it works before running full training
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Add src to path
sys.path.insert(0, 'src')

from tabular_dnn_classifier import TabularDNNClassifier

print("=" * 80)
print("Testing TabularDNNClassifier Implementation")
print("=" * 80)

# Generate synthetic data similar to BCC/SK dataset
print("\n1. Generating synthetic dataset (511 features, 1000 samples, imbalanced)...")
X, y = make_classification(
    n_samples=1000,
    n_features=511,
    n_informative=200,
    n_redundant=100,
    n_classes=2,
    weights=[0.84, 0.16],  # Similar to BCC/SK imbalance
    random_state=42,
    flip_y=0.05  # 5% noise
)

print(f"   Dataset shape: {X.shape}")
print(f"   Class distribution: {np.bincount(y)}")
print(f"   Class ratio: {np.bincount(y)[0]/np.bincount(y)[1]:.2f}:1")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

# Test with minimal epochs for quick validation
print("\n2. Training TabularDNNClassifier (quick test with 20 epochs)...")
dnn_classifier = TabularDNNClassifier(
    input_dim=511,
    hidden_units=[512, 384, 256, 128],  # Moderate architecture
    dropout_rate=0.3,
    l2_reg=1e-4,
    activation='swish',
    use_attention=True,
    learning_rate=1e-3,
    batch_size=64,
    epochs=20,  # Quick test
    patience=10,
    focal_loss_alpha=0.16,
    focal_loss_gamma=2.0,
    mixup_alpha=0.2,
    validation_split=0.2,
    verbose=1,  # Show progress
    random_state=42
)

print("\n   Starting training...")
dnn_classifier.fit(X_train_scaled, y_train)

# Evaluate
print("\n3. Evaluating on test set...")
y_pred = dnn_classifier.predict(X_test_scaled)
y_pred_proba = dnn_classifier.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n   Accuracy: {accuracy:.4f}")
print(f"   AUC Score: {auc_score:.4f}")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0 (Majority)', 'Class 1 (Minority)']))

# Check if model meets basic requirements
print("\n4. Validation checks:")
checks = []
checks.append(("Model trains without errors", True))
checks.append(("AUC > 0.50 (better than random)", auc_score > 0.50))
checks.append(("AUC > 0.70 (reasonable performance)", auc_score > 0.70))
checks.append(("Accuracy > 0.70", accuracy > 0.70))
checks.append(("Predictions have both classes", len(np.unique(y_pred)) == 2))
checks.append(("Probabilities in [0,1]", np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))))

all_passed = True
for check_name, passed in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"   {status}: {check_name}")
    if not passed:
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("SUCCESS: TabularDNNClassifier is working correctly!")
    print("You can now train it on your actual BCC/SK dataset.")
else:
    print("WARNING: Some checks failed. Review the model configuration.")
print("=" * 80)
