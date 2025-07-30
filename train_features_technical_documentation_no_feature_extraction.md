# Technical Documentation: BCC vs SK Classification Pipeline

## Executive Summary

This document presents a comprehensive technical analysis of the skin lesion classification system designed to differentiate between Basal Cell Carcinoma (BCC) and Seborrheic Keratosis (SK). The system employs advanced computer vision techniques, mathematical feature extraction, and machine learning algorithms to achieve clinical-grade diagnostic accuracy.

## 1. System Architecture Overview

The classification pipeline consists of six primary computational stages:

1. **Data Acquisition and Preprocessing**
2. **Lesion Segmentation and Mask Generation** 
3. **Multi-Domain Feature Extraction**
4. **Feature Space Optimization**
5. **Dimensionality Reduction and Selection**
6. **Multi-Classifier Ensemble Training**

### 1.1 Mathematical Foundation

The system operates on the principle of mapping dermoscopic images from pixel space to high-dimensional feature space:

```
f: I^(H×W×C) → R^n
```

Where:
- I = input image matrix
- H = image height in pixels
- W = image width in pixels  
- C = number of color channels
- R^n = n-dimensional real-valued feature space
- f = composite transformation function

## 2. Data Acquisition and Preprocessing Pipeline

### 2.1 Image Loading and Validation

**Algorithm 1: Multi-Format Image Loading**
```
INPUT: directory_path, supported_formats = {'.jpg', '.png', '.jpeg'}
OUTPUT: image_paths[], labels[]

FOR each file in directory_path:
    IF file.extension in supported_formats:
        image_paths.append(file.path)
        labels.append(directory_label)
    END IF
END FOR

RETURN image_paths, labels
```

### 2.2 Dataset Balancing Strategy

The system implements stratified sampling to ensure balanced representation:

```
N_bcc = min(|BCC_samples|, max_samples_per_class)
N_sk = min(|SK_samples|, max_samples_per_class)
```

Where:
- N_bcc = number of BCC samples selected
- N_sk = number of SK samples selected
- |BCC_samples| = total available BCC samples
- |SK_samples| = total available SK samples
- max_samples_per_class = user-defined maximum samples per class
- Balanced dataset size = min(N_bcc, N_sk) × 2

### 2.3 Label Encoding Scheme

Binary classification labels are encoded as:
- BCC (malignant): y = 1
- SK (benign): y = 0

Labels are explicitly cast to 32-bit integers to prevent floating-point conversion artifacts during training.

## 3. Advanced Lesion Segmentation

### 3.1 Transparency-Based Mask Generation

The system employs a threshold-based approach for lesion boundary detection:

**Algorithm 2: Lesion Mask Generation**
```
INPUT: image, threshold = 10
OUTPUT: binary_mask

alpha_channel = image[:,:,3] if image.channels == 4 else None
IF alpha_channel exists:
    mask = alpha_channel > threshold
ELSE:
    mask = generate_conventional_mask(image)
END IF

RETURN mask
```

### 3.2 Mathematical Formulation of Segmentation

The segmentation process can be expressed as:

```
M(x,y) = {
    1, if P(x,y) ∈ lesion_region
    0, otherwise
}
```

Where:
- M(x,y) = binary mask value at coordinates (x,y)
- P(x,y) = pixel intensity at coordinates (x,y)
- lesion_region = defined lesion boundary area

## 4. Feature Space Optimization

### 4.1 Feature Matrix Construction

The system constructs a feature matrix X ∈ R^(n×m) where:
- n = number of samples
- m = number of features

**Algorithm 3: Feature Matrix Assembly**
```
INPUT: feature_list[], expanded_keys[]
OUTPUT: feature_matrix X

X = zeros(len(feature_list), len(expanded_keys))

FOR i, features in enumerate(feature_list):
    FOR j, key in enumerate(expanded_keys):
        IF key contains array_index:
            base_key, index = parse_array_key(key)
            X[i,j] = features[base_key][index]
        ELSE:
            X[i,j] = features[key]
        END IF
    END FOR
END FOR

RETURN X
```

### 4.2 Data Cleaning and Validation

**Algorithm 4: Robust Feature Cleaning**
```
INPUT: feature_matrix X, feature_names[]
OUTPUT: cleaned_matrix X_clean, valid_features[]

// Remove infinite and NaN values
X_clean = replace_invalid_values(X, strategy='median')

// Remove zero-variance features
variance_mask = var(X_clean, axis=0) > threshold
X_clean = X_clean[:, variance_mask]
valid_features = feature_names[variance_mask]

// Validate numerical stability
assert not contains_invalid(X_clean)

RETURN X_clean, valid_features
```

## 5. Dimensionality Reduction and Feature Selection

### 5.1 Statistical Feature Selection Methods

#### 5.1.1 Mutual Information

```
MI(X,Y) = Σₓ Σᵧ p(x,y) × log₂[p(x,y)/(p(x)×p(y))]
```
Where:
- MI(X,Y) = mutual information between variables X and Y
- p(x,y) = joint probability distribution
- p(x),p(y) = marginal probability distributions
- log₂ = base-2 logarithm

#### 5.1.2 Chi-Square Test

```
χ² = Σᵢ [(Oᵢ - Eᵢ)²/Eᵢ]
```
Where:
- χ² = chi-square test statistic
- Oᵢ = observed frequency for category i
- Eᵢ = expected frequency for category i
- Σ = summation over all categories

#### 5.1.3 F-Score Analysis

```
F = [Σᶜ nᶜ(μᶜ - μ)²/(k-1)] / [Σᶜ Σᵢ∈ᶜ(xᵢ - μᶜ)²/(n-k)]
```
Where:
- F = F-statistic for ANOVA
- nᶜ = number of samples in class c
- μᶜ = mean of class c
- μ = overall mean
- k = number of classes
- n = total number of samples

### 5.2 Wrapper-Based Selection

**Recursive Feature Elimination (RFE):**

**Algorithm 5: RFE Implementation**
```
INPUT: X, y, estimator, n_features_to_select
OUTPUT: selected_features[], feature_ranking[]

current_features = all_features
WHILE len(current_features) > n_features_to_select:
    estimator.fit(X[:, current_features], y)
    importance = get_feature_importance(estimator)
    worst_feature = argmin(importance)
    current_features.remove(worst_feature)
END WHILE

RETURN current_features
```

## 6. Feature Scaling and Normalization

### 6.1 Robust Scaling Strategy

The system employs robust scaling to handle outliers:

```
X_scaled = (X - median(X)) / IQR(X)
```
Where:
- X_scaled = scaled feature values
- median(X) = median value of feature X
- IQR(X) = interquartile range (Q₃ - Q₁)

**Algorithm 6: Robust Scaling Implementation**
```
INPUT: X_train, X_test
OUTPUT: X_train_scaled, X_test_scaled

FOR each feature j:
    median_j = median(X_train[:, j])
    Q1_j = percentile(X_train[:, j], 25)
    Q3_j = percentile(X_train[:, j], 75)
    IQR_j = Q3_j - Q1_j
    
    X_train[:, j] = (X_train[:, j] - median_j) / IQR_j
    X_test[:, j] = (X_test[:, j] - median_j) / IQR_j
END FOR

RETURN X_train, X_test
```

## 7. Multi-Classifier Ensemble Framework

### 7.1 Algorithm Portfolio

The system implements diverse learning algorithms:

#### 7.1.1 Random Forest

**Ensemble Decision Function:**
```
ŷ = mode({h₁(x), h₂(x), ..., hₜ(x)})
```
Where:
- ŷ = predicted class label
- hᵢ(x) = prediction from i-th decision tree
- mode = most frequent prediction
- t = number of trees in ensemble

**Gini Impurity:**
```
Gini = 1 - Σᶜ pᶜ²
```
Where:
- Gini = Gini impurity measure
- pᶜ = proportion of samples belonging to class c
- Σ = summation over all classes

#### 7.1.2 Support Vector Machine

**Optimization Problem:**
```
min(w,b,ξ) ½||w||² + C Σᵢ ξᵢ

Subject to:
yᵢ(w·φ(xᵢ) + b) ≥ 1 - ξᵢ
ξᵢ ≥ 0
```
Where:
- w = weight vector
- b = bias term
- ξᵢ = slack variables
- C = regularization parameter
- φ(x) = feature mapping function

**RBF Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```
Where:
- K(xᵢ, xⱼ) = kernel function value
- γ = kernel parameter
- ||·|| = Euclidean distance

#### 7.1.3 Multi-Layer Perceptron

**Forward Propagation:**
```
aˡ⁺¹ = σ(Wˡaˡ + bˡ)
```
Where:
- aˡ = activation at layer l
- Wˡ = weight matrix at layer l
- bˡ = bias vector at layer l
- σ = activation function

**Backpropagation:**
```
∂L/∂Wˡ = aˡ⁻¹(δˡ)ᵀ
∂L/∂bˡ = δˡ
```
Where:
- L = loss function
- δˡ = error term at layer l
- δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)
- ⊙ = element-wise multiplication

#### 7.1.4 XGBoost

**Objective Function:**
```
L = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
```
Where:
- L = total loss function
- l(yᵢ, ŷᵢ) = loss for sample i
- Ω(fₖ) = regularization term for tree k
- Ω(f) = γT + ½λ||w||² (regularization)
- T = number of leaves
- λ = L2 regularization parameter

### 7.2 Hyperparameter Optimization

**Algorithm 7: Grid Search with Cross-Validation**
```
INPUT: parameter_grid, estimator, X, y, cv_folds
OUTPUT: best_params, best_score

best_score = -infinity
FOR each param_combination in parameter_grid:
    scores = []
    FOR each fold in cv_folds:
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = split(X, y, fold)
        estimator.set_params(param_combination)
        estimator.fit(X_train_fold, y_train_fold)
        score = evaluate(estimator, X_val_fold, y_val_fold)
        scores.append(score)
    END FOR
    
    mean_score = mean(scores)
    IF mean_score > best_score:
        best_score = mean_score
        best_params = param_combination
    END IF
END FOR

RETURN best_params, best_score
```

## 8. Performance Evaluation Framework

### 8.1 Classification Metrics

**Accuracy:**
Measures the overall correctness of the classifier across all classes.
This equation calculates the ratio of correct predictions to total predictions.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Where:
- TP = True Positives (correctly identified BCC cases)
- TN = True Negatives (correctly identified SK cases)
- FP = False Positives (SK cases incorrectly classified as BCC)
- FN = False Negatives (BCC cases incorrectly classified as SK)

**Precision:**
Quantifies the reliability of positive predictions by measuring how many predicted positive cases are actually positive.
This metric answers "Of all the cases predicted as BCC, how many are truly BCC?"
```
Precision = TP / (TP + FP)
```
Where:
- TP = True Positives
- FP = False Positives

**Recall (Sensitivity):**
Evaluates the classifier's ability to identify all actual positive cases in the dataset.
This metric answers "Of all the actual BCC cases, how many did we correctly identify?"
```
Recall = TP / (TP + FN)
```
Where:
- TP = True Positives
- FN = False Negatives

**F1-Score:**
Provides a balanced measure between precision and recall using their harmonic mean.
This composite metric is particularly useful when dealing with imbalanced datasets or when both false positives and false negatives are costly.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Where:
- Precision and Recall are as defined above
- Harmonic mean gives equal weight to both metrics

**ROC-AUC:**
Represents the area under the Receiver Operating Characteristic curve, measuring the classifier's ability to distinguish between classes.
This integral calculates the probability that the classifier ranks a randomly chosen positive instance higher than a randomly chosen negative instance.
```
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```
Where:
- TPR = True Positive Rate = TP/(TP+FN)
- FPR = False Positive Rate = FP/(FP+TN)
- FPR⁻¹(x) = inverse function of FPR
- ∫ = integral operator

### 8.2 Cross-Validation Strategy

**Stratified K-Fold Cross-Validation:**

**Algorithm 8: Stratified CV Implementation**
```
INPUT: X, y, k_folds
OUTPUT: cv_scores[]

FOR each fold i in range(k_folds):
    X_train, X_val, y_train, y_val = stratified_split(X, y, fold=i)
    
    // Maintain class distribution
    assert class_ratio(y_train) ≈ class_ratio(y_val)
    
    model.fit(X_train, y_train)
    score = evaluate(model, X_val, y_val)
    cv_scores.append(score)
END FOR

RETURN cv_scores
```

## 9. Implementation Specifications

### 9.1 Computational Complexity

**Feature Extraction Complexity:**
- Geometric features: O(n²) where n is boundary length
- Color features: O(HW) where H×W is image dimensions
- Texture features: O(HW × f) where f is filter count
- GLCM computation: O(HW × d × θ) for d distances and θ orientations

**Training Complexity:**
- Random Forest: O(M × N × log(N) × F) where M=trees, N=samples, F=features
- SVM: O(N² × F) to O(N³ × F) depending on kernel
- Neural Network: O(epochs × N × H × F) where H=hidden units

### 9.2 Memory Requirements

**Feature Matrix Storage:**
```
Memory = N × F × 8 bytes (double precision)
```
Where:
- N = number of samples
- F = number of features
- 8 bytes = storage per double precision value

For typical datasets: 1000 samples × 400 features = 3.2 MB

### 9.3 Error Handling and Robustness

**Algorithm 9: Robust Processing Pipeline**
```
INPUT: image_path
OUTPUT: features or error_status

TRY:
    image = load_image(image_path)
    validate_image_format(image)
    
    mask = generate_lesion_mask(image)
    validate_mask_quality(mask)
    
    features = extract_all_features(image, mask)
    validate_feature_completeness(features)
    
    RETURN features
    
CATCH exception:
    log_error(exception, image_path)
    RETURN empty_feature_dict()
END TRY
```

## 10. Clinical Validation Framework

### 10.1 Medical Relevance Assessment

The feature extraction framework aligns with established dermatological criteria:

**Dermoscopic Pattern Analysis:**
- Vascular patterns (BCC indicators)
- Pigment network structures
- Border definition characteristics
- Color distribution patterns

**Clinical Decision Support:**
- Feature interpretability for medical professionals
- Confidence scoring for diagnostic assistance
- Uncertainty quantification for edge cases

### 10.2 Validation Metrics

**Clinical Sensitivity:**
```
Sensitivity = True Positive Rate = TP / (TP + FN)
```
Where:
- Sensitivity = proportion of actual BCC cases correctly identified
- TP = True Positives (BCC correctly identified as BCC)
- FN = False Negatives (BCC incorrectly identified as SK)

**Clinical Specificity:**
```
Specificity = True Negative Rate = TN / (TN + FP)
```
Where:
- Specificity = proportion of actual SK cases correctly identified
- TN = True Negatives (SK correctly identified as SK)
- FP = False Positives (SK incorrectly identified as BCC)

**Positive Predictive Value:**
```
PPV = TP / (TP + FP)
```
Where:
- PPV = probability that positive prediction is correct
- TP = True Positives
- FP = False Positives

**Negative Predictive Value:**
```
NPV = TN / (TN + FN)
```
Where:
- NPV = probability that negative prediction is correct
- TN = True Negatives
- FN = False Negatives