"""
Script to extract feature names from a trained model and create metadata file
"""
from joblib import load
import json
import numpy as np
from src.conventional_features import ConventionalFeatureExtractor
from PIL import Image
import cv2

# Initialize feature extractor
extractor = ConventionalFeatureExtractor()

# Create a dummy image to extract feature names
dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
dummy_mask = np.ones((224, 224), dtype=bool)

# Extract features to get feature names
features_dict = extractor.extract_all_features(dummy_image, dummy_mask)

# Convert to flat list of feature names
feature_names = []
for key, value in features_dict.items():
    if isinstance(value, (list, np.ndarray)):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            feature_names.append(key)
        else:
            for i in range(len(value)):
                feature_names.append(f"{key}_{i}")
    else:
        feature_names.append(key)

print(f"Total features from current extractor: {len(feature_names)}")

# Load the selector to see what it expects
selector = load('model/feature_based/highest_model_svm_rbf/selector.joblib')
print(f"Selector expects: {selector.n_features_in_} features")
print(f"Selected features: {selector.get_support().sum()}")

# Get selected feature indices
selected_indices = selector.get_support(indices=True)

# If we have more features now than training, we need to handle this
if len(feature_names) > selector.n_features_in_:
    print(f"\nWARNING: Current extractor produces {len(feature_names)} features")
    print(f"But model was trained on {selector.n_features_in_} features")
    print(f"Difference: {len(feature_names) - selector.n_features_in_} extra features")
    
    # Use only the first n features that match training
    feature_names_used = feature_names[:selector.n_features_in_]
elif len(feature_names) < selector.n_features_in_:
    print(f"\nWARNING: Current extractor produces only {len(feature_names)} features")
    print(f"But model was trained on {selector.n_features_in_} features")
    print(f"Missing: {selector.n_features_in_ - len(feature_names)} features")
    feature_names_used = feature_names
else:
    feature_names_used = feature_names

# Save metadata
metadata = {
    "n_features_expected": selector.n_features_in_,
    "n_features_selected": len(selected_indices),
    "feature_names": feature_names_used,
    "selected_feature_indices": selected_indices.tolist(),
    "selected_feature_names": [feature_names_used[i] for i in selected_indices if i < len(feature_names_used)]
}

output_path = 'model/feature_based/highest_model_svm_rbf/feature_metadata.json'
with open(output_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved to: {output_path}")
print(f"Feature names saved: {len(metadata['feature_names'])}")
print(f"Selected feature names: {len(metadata['selected_feature_names'])}")
