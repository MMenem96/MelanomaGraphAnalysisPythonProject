from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os
import uuid
import time
import glob
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.preprocessing import ImagePreprocessor
from src.conventional_features import ConventionalFeatureExtractor
from src.image_validator import ImageValidator
from src.segmentation.skin_lesion_processor import SkinLesionProcessor
from joblib import load
import cv2
import numpy as np
import logging
from datetime import datetime
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Set up directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
MODEL_FOLDER = "model"
DATA_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "bcc"), exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "bkl"), exist_ok=True)

# Define routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page with information on the BCC vs BKL detection method using conventional feature engineering."""
    return render_template('about.html')
    
# Remove comparison route to disable web display of model comparisons

# History functionality removed

# Training image upload functionality removed - only available through manual CLI tools

@app.route('/train')
def train():
    """Inform users that models can only be trained via the command line."""
    flash("Training functionality is only available via the command line using manual_run_main.py. Please contact the administrator for model training.", "info")
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded image and perform BCC vs BKL detection using conventional feature engineering."""
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            flash("No image selected", "danger")
            return redirect(url_for('index'))
        
        image_file = request.files['image']
        if image_file.filename == '':
            flash("No image selected", "danger")
            return redirect(url_for('index'))
        
        # Get the selected classifier type
        classifier_type = request.form.get('classifier_type', 'random_forest')
        
        # Map the form value to the actual model folder name
        classifier_map = {
            'random_forest': 'RF',
            'svm_rbf': 'SVM (RBF)',
            'xgboost': 'XGBoost',
            'gradient_boosting': 'Gradient Boosting',
            'knn': 'KNN',
            'mlp': 'MLP',
            'logistic_regression': 'Logistic Regression'
        }
        
        # Use the mapped classifier type
        model_name = classifier_map.get(classifier_type, 'RF')
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename and path
        original_filename = image_file.filename
        filename = f"{timestamp}_{original_filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        
        # Initialize components
        preprocessor = ImagePreprocessor()
        segmenter = SkinLesionProcessor()
        feature_extractor = ConventionalFeatureExtractor()
        image_validator = ImageValidator()
        
        # Load and validate image
        try:
            # Load with PIL for transparency support
            pil_image = Image.open(image_path)
            if pil_image.mode == 'RGBA':
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])
                original_image = np.array(background)
            else:
                original_image = np.array(pil_image.convert('RGB'))
            
            # Validation disabled for pre-segmented dermoscopic images
            # is_valid, validation_message = image_validator.validate_skin_image(original_image)
            # 
            # if not is_valid:
            #     flash(f"Invalid skin image: {validation_message}", "danger")
            #     return redirect(url_for('index'))
        except Exception as e:
            app.logger.error(f"Error loading image: {str(e)}")
            flash(f"Error loading image: {str(e)}", "danger")
            return redirect(url_for('index'))
        
        # Process image
        try:
            # Apply preprocessing pipeline (hair removal, gaussian filtering)
            grayscale_image = segmenter.convert_to_grayscale(original_image)
            combined_hair_mask, _, _ = segmenter.apply_combined_hair_detection(grayscale_image)
            inpainted_image = segmenter.apply_inpainting(original_image, combined_hair_mask)
            processed_image = segmenter.apply_gaussian_blur(inpainted_image)
            
            # Save preprocessed image
            preprocessed_image_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}_preprocessed.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(processed_image)
            plt.axis('off')
            plt.title('Preprocessed Image (Hair Removal + Gaussian Filter)')
            plt.tight_layout()
            plt.savefig(preprocessed_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Generate lesion mask from transparent background
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 4:
                alpha_channel = processed_image[:, :, 3]
                lesion_mask = alpha_channel > 10
            else:
                gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                lesion_mask = gray < 245
            
            # Extract conventional features (color, texture, geometric, Krawtchouk moments)
            extracted_features = feature_extractor.extract_all_features(processed_image, lesion_mask)
            
            app.logger.info(f"Extracted {len(extracted_features)} feature types from image")
            
            # Use the highest performing SVM RBF model (highest_model_svm_rbf)
            feature_based_dir = os.path.join(MODEL_FOLDER, 'feature_based')
            model_dir = os.path.join(feature_based_dir, 'highest_model_svm_rbf')
            
            # Verify the model directory exists
            if not os.path.exists(model_dir):
                app.logger.error(f"Highest performing SVM RBF model not found at: {model_dir}")
                flash(f"Best model not found. Please ensure highest_model_svm_rbf exists in model/feature_based/", "warning")
                return redirect(url_for('index'))
            
            model_path = os.path.join(model_dir, 'model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            selector_path = os.path.join(model_dir, 'selector.joblib')
            config_path = os.path.join(model_dir, 'classifier_config.json')
            metadata_path = os.path.join(model_dir, 'feature_metadata.json')
            
            app.logger.info(f"Loading highest performing SVM RBF model from: {model_dir}")
            
            # Load model components
            model = load(model_path)
            scaler = load(scaler_path)
            selector = load(selector_path) if os.path.exists(selector_path) else None
            
            # Load config for model info
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('name', 'SVM (RBF)')
            else:
                model_name = 'SVM (RBF)'
            
            # Load feature metadata for proper feature alignment
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                expected_feature_names = metadata.get('feature_names', [])
                app.logger.info(f"Using feature metadata with {len(expected_feature_names)} expected features")
            else:
                # Fallback: flatten extracted features
                expected_feature_names = []
                for key, value in extracted_features.items():
                    if isinstance(value, (list, np.ndarray)):
                        if isinstance(value, np.ndarray) and value.ndim == 0:
                            expected_feature_names.append(key)
                        else:
                            for i in range(len(value)):
                                expected_feature_names.append(f"{key}_{i}")
                    else:
                        expected_feature_names.append(key)
                app.logger.warning(f"No metadata found, using {len(expected_feature_names)} features from current extraction")
            
            # Convert extracted features to array matching expected features
            X_raw = []
            for fname in expected_feature_names:
                # Handle both direct keys and indexed keys (e.g., 'color_mean_0')
                if fname in extracted_features:
                    val = extracted_features[fname]
                    if isinstance(val, (list, np.ndarray)):
                        if isinstance(val, np.ndarray) and val.ndim == 0:
                            X_raw.append(float(val))
                        else:
                            X_raw.extend([float(v) for v in val])
                    else:
                        X_raw.append(float(val))
                elif '_' in fname:
                    # Handle indexed feature names like 'color_mean_0'
                    base_name = fname.rsplit('_', 1)[0]
                    if base_name in extracted_features:
                        val = extracted_features[base_name]
                        if isinstance(val, (list, np.ndarray)):
                            try:
                                idx = int(fname.rsplit('_', 1)[1])
                                X_raw.append(float(val[idx]))
                            except (ValueError, IndexError):
                                X_raw.append(0.0)
                        else:
                            X_raw.append(0.0)
                    else:
                        X_raw.append(0.0)
                else:
                    X_raw.append(0.0)  # Missing feature
            
            X_raw = np.array(X_raw).reshape(1, -1)
            app.logger.info(f"Feature array shape after alignment: {X_raw.shape}")
            
            # Apply feature selection if selector exists
            if selector is not None:
                X_selected = selector.transform(X_raw)
            else:
                X_selected = X_raw
            
            # Scale features
            X_scaled = scaler.transform(X_selected)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Get probability (handle different classifier types)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X_scaled)[0][1]
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X_scaled)[0]
                probability = 1 / (1 + np.exp(-decision))  # Sigmoid transform
            else:
                probability = float(prediction)
            
            app.logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
            
            prediction_label = "Basal Cell Carcinoma (BCC)" if prediction == 1 else "Benign Keratosis-like Lesion (BKL)"
            probability_pct = probability * 100
            
            # Determine risk level based on probability
            if probability_pct > 75:
                risk_level = "HIGH"
                explanation = "High probability of Basal Cell Carcinoma (BCC). Immediate dermatological consultation recommended for biopsy and treatment planning."
            elif probability_pct > 50:
                risk_level = "MODERATE TO HIGH"
                explanation = "Elevated probability of Basal Cell Carcinoma (BCC). Prompt medical evaluation recommended to confirm diagnosis."
            elif probability_pct > 25:
                risk_level = "MODERATE"
                explanation = "Some features suggesting BCC are present. Medical evaluation advised for professional assessment."
            else:
                risk_level = "LOW"
                explanation = "Low probability of BCC. Likely Benign Keratosis-like Lesion (BKL). Continue regular skin monitoring and self-examination."
            
            # Store result with enhanced information
            result = {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'original_filename': original_filename,
                'image_path': image_path,
                'preprocessed_image_path': preprocessed_image_path,
                'prediction': prediction_label,
                'probability': probability,
                'probability_pct': probability_pct,
                'risk_level': risk_level,
                'explanation': explanation,
                'model_used': model_name,
                'classifier_type': classifier_type,
                'features_extracted': len(extracted_features),
                'features_used': X_scaled.shape[1]
            }
            
            # In a real app, you'd store this in a database
            
            return render_template('results.html', 
                                 result=result,
                                 probability_percent=int(probability*100))
        
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            flash(f"Error processing image: {str(e)}", "danger")
            return redirect(url_for('index'))
    
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for BCC vs BKL image analysis using conventional feature engineering."""
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({"status": 400,
                            "message": "No image provided"
                            }
                           ), 400

        image_file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, "temp_api.jpg")
        image_file.save(image_path)

        # Initialize components
        segmenter = SkinLesionProcessor()
        feature_extractor = ConventionalFeatureExtractor()
        image_validator = ImageValidator()

        # Load image with PIL for transparency support
        pil_image = Image.open(image_path)
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            original_image = np.array(background)
        else:
            original_image = np.array(pil_image.convert('RGB'))
        
        # Validation disabled for pre-segmented dermoscopic images
        # is_valid, validation_message = image_validator.validate_skin_image(original_image)
        # 
        # if not is_valid:
        #     return jsonify({
        #         "status": 400,
        #         "message": "Invalid input image",
        #         "details": validation_message
        #     }), 400

        # Apply preprocessing pipeline
        grayscale_image = segmenter.convert_to_grayscale(original_image)
        combined_hair_mask, _, _ = segmenter.apply_combined_hair_detection(grayscale_image)
        inpainted_image = segmenter.apply_inpainting(original_image, combined_hair_mask)
        processed_image = segmenter.apply_gaussian_blur(inpainted_image)
        
        # Generate lesion mask
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 4:
            alpha_channel = processed_image[:, :, 3]
            lesion_mask = alpha_channel > 10
        else:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            lesion_mask = gray < 245
        
        # Extract conventional features
        extracted_features = feature_extractor.extract_all_features(processed_image, lesion_mask)

        # Use the highest performing SVM RBF model (highest_model_svm_rbf) - ignore classifier_type parameter
        feature_based_dir = os.path.join(MODEL_FOLDER, 'feature_based')
        model_dir = os.path.join(feature_based_dir, 'highest_model_svm_rbf')
        
        if not os.path.exists(model_dir):
            return jsonify({
                "status": 500,
                "message": f"Highest performing SVM RBF model not found. Please ensure highest_model_svm_rbf exists in model/feature_based/"
            }), 500
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        selector_path = os.path.join(model_dir, 'selector.joblib')
        config_path = os.path.join(model_dir, 'classifier_config.json')
        metadata_path = os.path.join(model_dir, 'feature_metadata.json')
        
        app.logger.info(f"API: Loading highest performing SVM RBF model from: {model_dir}")
        
        # Load model components
        model = load(model_path)
        scaler = load(scaler_path)
        selector = load(selector_path) if os.path.exists(selector_path) else None
        
        # Load config for model info
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_name = config.get('name', 'SVM (RBF)')
        else:
            model_name = 'SVM (RBF)'
        
        # Load feature metadata for proper feature alignment
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            expected_feature_names = metadata.get('feature_names', [])
        else:
            # Fallback: flatten extracted features
            expected_feature_names = []
            for key, value in extracted_features.items():
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        expected_feature_names.append(key)
                    else:
                        for i in range(len(value)):
                            expected_feature_names.append(f"{key}_{i}")
                else:
                    expected_feature_names.append(key)
        
        # Convert features to array matching expected features
        X_raw = []
        for fname in expected_feature_names:
            if fname in extracted_features:
                val = extracted_features[fname]
                if isinstance(val, (list, np.ndarray)):
                    if isinstance(val, np.ndarray) and val.ndim == 0:
                        X_raw.append(float(val))
                    else:
                        X_raw.extend([float(v) for v in val])
                else:
                    X_raw.append(float(val))
            elif '_' in fname:
                base_name = fname.rsplit('_', 1)[0]
                if base_name in extracted_features:
                    val = extracted_features[base_name]
                    if isinstance(val, (list, np.ndarray)):
                        try:
                            idx = int(fname.rsplit('_', 1)[1])
                            X_raw.append(float(val[idx]))
                        except (ValueError, IndexError):
                            X_raw.append(0.0)
                    else:
                        X_raw.append(0.0)
                else:
                    X_raw.append(0.0)
            else:
                X_raw.append(0.0)
        
        X_raw = np.array(X_raw).reshape(1, -1)
        
        # Apply feature selection and scaling
        if selector is not None:
            X_selected = selector.transform(X_raw)
        else:
            X_selected = X_raw
        
        X_scaled = scaler.transform(X_selected)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X_scaled)[0][1]
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(X_scaled)[0]
            probability = 1 / (1 + np.exp(-decision))
        else:
            probability = float(prediction)
        
        app.logger.info(f"API Prediction: {prediction}, Probability: {probability:.4f}")
        
        prediction_label = "Basal Cell Carcinoma (BCC)" if prediction == 1 else "Benign Keratosis-like Lesion (BKL)"
        probability_pct = probability * 100
        
        # Determine risk level
        if probability_pct > 75:
            risk_level = "HIGH"
            explanation = "High probability of Basal Cell Carcinoma (BCC). Immediate dermatological consultation recommended for biopsy and treatment planning."
        elif probability_pct > 50:
            risk_level = "MODERATE TO HIGH"
            explanation = "Elevated probability of Basal Cell Carcinoma (BCC). Prompt medical evaluation recommended to confirm diagnosis."
        elif probability_pct > 25:
            risk_level = "MODERATE"
            explanation = "Some features suggesting BCC are present. Medical evaluation advised for professional assessment."
        else:
            risk_level = "LOW"
            explanation = "Low probability of BCC. Likely Benign Keratosis-like Lesion (BKL). Continue regular skin monitoring and self-examination."

        return jsonify({
            "status": 200,
            "message": "Image processed successfully",
            "data": {
                "prediction": prediction_label,
                "probability": float(probability),
                "probability_percent": float(probability_pct),
                "risk_level": risk_level,
                "explanation": explanation,
                "model_used": model_name,
                "features_extracted": len(extracted_features),
                "features_used": int(X_scaled.shape[1])
            }}), 200

    except Exception as e:
        app.logger.error(f"Error processing API request: {str(e)}")
        return jsonify({"status": 500, "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)