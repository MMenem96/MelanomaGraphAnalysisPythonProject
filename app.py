from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os
import uuid
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
from src.conventional_features import ConventionalFeatureExtractor
from src.classifier import BCCSKClassifier
from src.image_validator import ImageValidator
from src.visualization import Visualizer
from joblib import load, dump
import cv2
import numpy as np
import logging
from datetime import datetime
from skimage.segmentation import mark_boundaries
import networkx as nx

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
os.makedirs(os.path.join(DATA_FOLDER, "sk"), exist_ok=True)

# Define routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page with information on the BCC vs SK detection method."""
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
    """Process uploaded image and perform BCC vs SK detection."""
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
        classifier_type = request.form.get('classifier_type', 'svm_rbf')
        
        # Map the form value to the actual classifier type
        classifier_map = {
            'svm_rbf': 'svm',  # Default SVM uses RBF kernel
            'svm_sigmoid': 'svm_sigmoid',
            'svm_poly': 'svm_poly',
            'rf': 'rf',
            'knn': 'knn',
            'mlp': 'mlp'
        }
        
        # Use the mapped classifier type or default to SVM
        actual_classifier_type = classifier_map.get(classifier_type, 'svm')
        
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
        superpixel_gen = SuperpixelGenerator(n_segments=20, compactness=10)
        graph_constructor = GraphConstructor(connectivity_threshold=0.5)
        feature_extractor = FeatureExtractor()
        conv_feature_extractor = ConventionalFeatureExtractor()
        classifier = BCCSKClassifier(classifier_type=actual_classifier_type)
        image_validator = ImageValidator()
        visualizer = Visualizer()
        
        # Load and validate image
        try:
            original_image = preprocessor.load_image(image_path)
            is_valid, validation_message = image_validator.validate_skin_image(original_image)
            
            if not is_valid:
                flash(f"Invalid skin image: {validation_message}", "danger")
                return redirect(url_for('index'))
        except Exception as e:
            app.logger.error(f"Error loading image: {str(e)}")
            flash(f"Error loading image: {str(e)}", "danger")
            return redirect(url_for('index'))
        
        # Process image
        try:
            # Preprocess image
            processed_image = preprocessor.preprocess(original_image)
            
            # Generate superpixels
            segments = superpixel_gen.generate_superpixels(processed_image)
            features = superpixel_gen.compute_superpixel_features(processed_image, segments)
            
            # Visualize superpixels
            superpixels_image_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}_superpixels.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(mark_boundaries(processed_image, segments))
            plt.axis('off')
            plt.title('Superpixel Segmentation')
            plt.tight_layout()
            plt.savefig(superpixels_image_path)
            plt.close()
            
            # Construct graph
            G = graph_constructor.build_graph(features, segments)
            
            # Extract graph-based features
            G.graph['features'] = {
                **feature_extractor.extract_local_features(G),
                **feature_extractor.extract_global_features(G),
                **feature_extractor.extract_spectral_features(G)
            }
            
            # Visualize graph
            graph_image_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}_graph.png")
            visualizer.plot_graph(G)
            
            # Calculate mask of the lesion
            lesion_mask = segments > -1
            
            # Extract conventional image features
            conventional_features = conv_feature_extractor.extract_all_features(original_image, lesion_mask)
            G.graph['conventional_features'] = conventional_features
            
            # Determine the model path based on classifier type
            if classifier_type == 'svm_rbf':
                model_dir = os.path.join(MODEL_FOLDER, 'SVM_RBF')
            elif classifier_type == 'svm_sigmoid':
                model_dir = os.path.join(MODEL_FOLDER, 'SVM_Sigmoid')
            elif classifier_type == 'svm_poly':
                model_dir = os.path.join(MODEL_FOLDER, 'SVM_Poly')
            elif classifier_type == 'rf':
                model_dir = os.path.join(MODEL_FOLDER, 'RF')
            elif classifier_type == 'knn':
                model_dir = os.path.join(MODEL_FOLDER, 'KNN')
            elif classifier_type == 'mlp':
                model_dir = os.path.join(MODEL_FOLDER, 'MLP')
            else:
                model_dir = os.path.join(MODEL_FOLDER, 'SVM_RBF')  # Default
            
            # Check if specific model exists
            model_path = os.path.join(model_dir, 'model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            feature_selector_path = os.path.join(model_dir, 'feature_selector.joblib')
            
            # If the specific model doesn't exist, fall back to default model
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                app.logger.warning(f"Model {classifier_type} not found, using default model instead")
                model_path = os.path.join(MODEL_FOLDER, 'bcc_sk_classifier.joblib')
                scaler_path = os.path.join(MODEL_FOLDER, 'scaler.joblib')
                feature_selector_path = os.path.join(MODEL_FOLDER, 'feature_selector.joblib')
                
                # If even the default model doesn't exist, prompt to train
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    flash("No trained models found. Please train the models first.", "warning")
                    return redirect(url_for('train'))
            
            # Load model and scaler
            model = load(model_path)
            scaler = load(scaler_path)
            
            app.logger.info(f"Using model: {model_path}")
            
            # Check if feature selector exists and load it
            feature_selector = None
            if os.path.exists(feature_selector_path):
                app.logger.info(f"Using feature selector: {feature_selector_path}")
                feature_selector = load(feature_selector_path)
            
            # Prepare features for prediction
            X = classifier.prepare_features([G])
            X_scaled = scaler.transform(X)
            
            # Apply feature selection if available
            if feature_selector is not None:
                X_selected = feature_selector.transform(X_scaled)
                app.logger.info(f"Feature selection applied: {X.shape[1]} features reduced to {X_selected.shape[1]}")
                
                # Make prediction with selected features
                prediction = model.predict(X_selected)[0]
                probability = model.predict_proba(X_selected)[0][1] if hasattr(model, 'predict_proba') else 0.5
            else:
                # Make prediction without feature selection
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
            
            prediction_label = "Basal-cell Carcinoma (BCC)" if prediction == 1 else "Seborrheic Keratosis (SK)"
            probability_pct = probability * 100
            
            # Determine risk level based on probability
            if probability_pct > 75:
                risk_level = "HIGH"
                explanation = "High probability of Basal-cell Carcinoma (BCC). Immediate medical consultation recommended."
            elif probability_pct > 50:
                risk_level = "MODERATE TO HIGH"
                explanation = "Elevated probability of Basal-cell Carcinoma (BCC). Prompt medical consultation recommended."
            elif probability_pct > 25:
                risk_level = "MODERATE"
                explanation = "Some features suggesting BCC are present. Medical evaluation advised."
            else:
                risk_level = "LOW"
                explanation = "Low probability of BCC. Likely Seborrheic Keratosis (SK). Regular self-examination advised."
            
            # Store result with enhanced information
            result = {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'original_filename': original_filename,
                'image_path': image_path,
                'superpixels_image_path': superpixels_image_path,
                'graph_image_path': graph_image_path,
                'prediction': prediction_label,
                'probability': probability,
                'probability_pct': probability_pct,
                'risk_level': risk_level,
                'explanation': explanation,
                'validation_message': validation_message,
                'features_extracted': len(X[0]),
                'features_selected': X_selected.shape[1] if feature_selector is not None else None
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
    """API endpoint for image analysis."""
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            return jsonify({"status": 400,
                            "message": "No image provided"
                            }
                           ), 400

        image_file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, "temp.jpg")
        image_file.save(image_path)

        # Process Image
        preprocessor = ImagePreprocessor()
        superpixel_gen = SuperpixelGenerator(n_segments=20, compactness=10)
        graph_constructor = GraphConstructor(connectivity_threshold=0.5)
        feature_extractor = FeatureExtractor()
        conv_feature_extractor = ConventionalFeatureExtractor()
        classifier = BCCSKClassifier(classifier_type='svm')
        image_validator = ImageValidator()

        # Load original image for processing
        original_image = preprocessor.load_image(image_path)
        
        # Validate if this is a skin lesion image
        is_valid, validation_message = image_validator.validate_skin_image(original_image)
        
        if not is_valid:
            # Return error for invalid skin images
            return jsonify({
                "status": 400,
                "message": "Invalid input image",
                "details": validation_message
            }), 400

        # Continue with regular processing if image is valid
        processed_image = preprocessor.preprocess(original_image)

        # Generate superpixels
        segments = superpixel_gen.generate_superpixels(processed_image)
        features = superpixel_gen.compute_superpixel_features(processed_image, segments)

        # Construct graph
        G = graph_constructor.build_graph(features, segments)

        # Extract graph-based features
        G.graph['features'] = {
            **feature_extractor.extract_local_features(G),
            **feature_extractor.extract_global_features(G),
            **feature_extractor.extract_spectral_features(G)
        }
        
        # Calculate mask of the lesion (combining all superpixels)
        lesion_mask = segments > -1  # All superpixels are part of the lesion

        # Extract conventional image features
        conventional_features = conv_feature_extractor.extract_all_features(original_image, lesion_mask)
        
        # Store conventional features in the graph
        G.graph['conventional_features'] = conventional_features

        # Get the classifier type from query params or default to SVM_RBF
        classifier_type = request.args.get('classifier_type', 'svm_rbf')
        
        # Determine the model path based on classifier type
        if classifier_type == 'svm_rbf':
            model_dir = os.path.join(MODEL_FOLDER, 'SVM_RBF')
        elif classifier_type == 'svm_sigmoid':
            model_dir = os.path.join(MODEL_FOLDER, 'SVM_Sigmoid')
        elif classifier_type == 'svm_poly':
            model_dir = os.path.join(MODEL_FOLDER, 'SVM_Poly')
        elif classifier_type == 'rf':
            model_dir = os.path.join(MODEL_FOLDER, 'RF')
        elif classifier_type == 'knn':
            model_dir = os.path.join(MODEL_FOLDER, 'KNN')
        elif classifier_type == 'mlp':
            model_dir = os.path.join(MODEL_FOLDER, 'MLP')
        else:
            model_dir = os.path.join(MODEL_FOLDER, 'SVM_RBF')  # Default
        
        # Check if specific model exists
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        feature_selector_path = os.path.join(model_dir, 'feature_selector.joblib')
        
        # If the specific model doesn't exist, fall back to default model
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            app.logger.warning(f"Model {classifier_type} not found in API call, using default model instead")
            model_path = os.path.join(MODEL_FOLDER, 'bcc_sk_classifier.joblib')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler.joblib')
            feature_selector_path = os.path.join(MODEL_FOLDER, 'feature_selector.joblib')
            
            # If even the default model doesn't exist, return an error
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return jsonify({"error": "Model not found. Train the model first."}), 500

        model = load(model_path)
        scaler = load(scaler_path)
        
        # Load feature selector if it exists
        feature_selector = None
        if os.path.exists(feature_selector_path):
            feature_selector = load(feature_selector_path)

        # Prepare features for prediction
        X = classifier.prepare_features([G])
        X_scaled = scaler.transform(X)
        
        # Apply feature selection if available
        if feature_selector is not None:
            try:
                X_selected = feature_selector.transform(X_scaled)
                app.logger.info(f"Feature selection applied: {X.shape[1]} features reduced to {X_selected.shape[1]}")
                
                # Make prediction with selected features
                prediction = model.predict(X_selected)[0]
                probability = model.predict_proba(X_selected)[0][1] if hasattr(model, 'predict_proba') else 0.5
            except Exception as fs_error:
                app.logger.warning(f"Error applying feature selection: {str(fs_error)}. Using full feature set.")
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
        else:
            # Make prediction without feature selection
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        prediction_label = "Basal-cell Carcinoma (BCC)" if prediction == 1 else "Seborrheic Keratosis (SK)"
        probability_pct = probability * 100
        
        # Determine risk level based on probability
        explanation = ""
        if probability_pct > 75:
            risk_level = "HIGH"
            explanation = "High probability of Basal-cell Carcinoma (BCC). Immediate medical consultation recommended."
        elif probability_pct > 50:
            risk_level = "MODERATE TO HIGH"
            explanation = "Elevated probability of Basal-cell Carcinoma (BCC). Prompt medical consultation recommended."
        elif probability_pct > 25:
            risk_level = "MODERATE"
            explanation = "Some features suggesting BCC are present. Medical evaluation advised."
        else:
            risk_level = "LOW"
            explanation = "Low probability of BCC. Likely Seborrheic Keratosis (SK). Regular self-examination advised."

        return jsonify({
            "status": 200,
            "message": "Image processed successfully",
            "data": {
                "prediction": prediction_label,
                "probability": float(probability),
                "probability_percent": float(probability_pct),
                "risk_level": risk_level,
                "explanation": explanation,
                "validation": validation_message,
                "features_extracted": len(X[0]),
                "features_used": X_selected.shape[1] if feature_selector is not None and 'X_selected' in locals() else len(X[0])
            }}), 200

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": 500, "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)