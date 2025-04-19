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
from src.classifier import MelanomaClassifier
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
os.makedirs(os.path.join(DATA_FOLDER, "melanoma"), exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "benign"), exist_ok=True)

# Define routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page with information on the melanoma detection method."""
    return render_template('about.html')

@app.route('/history')
def history():
    """Render the history page showing past analyses and training records."""
    # This is a placeholder. In a real app, you'd retrieve actual records
    analyses = [
        {
            'id': '123456',
            'date': '2025-04-18',
            'result': 'Benign',
            'confidence': 0.89
        }
    ]
    return render_template('history.html', analyses=analyses)

@app.route('/upload_training_images', methods=['POST'])
def upload_training_images():
    """Upload training images for model building."""
    if 'images' not in request.files:
        flash("No images selected", "danger")
        return redirect(url_for('train'))
    
    # Get the image type (melanoma or benign)
    image_type = request.form.get('image_type', '')
    if image_type not in ['melanoma', 'benign']:
        flash("Invalid image type", "danger")
        return redirect(url_for('train'))
    
    # Define the save directory
    save_dir = os.path.join(DATA_FOLDER, image_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # Process and save multiple uploaded files
    files = request.files.getlist('images')
    saved_count = 0
    
    for file in files:
        if file and file.filename:
            # Create a secure filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(save_dir, filename)
            
            try:
                # Save the file
                file.save(file_path)
                saved_count += 1
                
                # Validate the image
                validator = ImageValidator()
                image = cv2.imread(file_path)
                
                if image is None:
                    os.remove(file_path)
                    app.logger.warning(f"Could not read image: {filename}")
                    continue
                    
                is_valid, message = validator.validate_skin_image(image)
                
                if not is_valid:
                    os.remove(file_path)
                    app.logger.warning(f"Invalid image removed: {filename}, reason: {message}")
                    continue
                    
            except Exception as e:
                app.logger.error(f"Error saving file {filename}: {str(e)}")
                flash(f"Error saving file {filename}: {str(e)}", "danger")
    
    flash(f"Successfully uploaded {saved_count} {image_type} images.", "success")
    return redirect(url_for('train'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train the classifier with uploaded images."""
    if request.method == 'POST':
        try:
            # Get training parameters
            n_segments = int(request.form.get('n_segments', 20))
            compactness = float(request.form.get('compactness', 10.0))
            connectivity_threshold = float(request.form.get('connectivity_threshold', 0.5))
            classifier_type = request.form.get('classifier', 'svm')
            
            # Initialize dataset handler
            from src.dataset_handler import DatasetHandler
            dataset_handler = DatasetHandler(
                n_segments=n_segments,
                compactness=compactness,
                connectivity_threshold=connectivity_threshold
            )
            
            # Process dataset
            app.logger.info("Processing dataset...")
            melanoma_dir = os.path.join(DATA_FOLDER, "melanoma")
            benign_dir = os.path.join(DATA_FOLDER, "benign")
            
            # Check if directories contain images
            melanoma_files = [f for f in os.listdir(melanoma_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            benign_files = [f for f in os.listdir(benign_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(melanoma_files) == 0 or len(benign_files) == 0:
                flash("Please upload training images first. You need at least one melanoma and one benign image.", "danger")
                return redirect(url_for('train'))
            
            graphs, labels = dataset_handler.process_dataset(
                melanoma_dir,
                benign_dir
            )
            
            # Split dataset
            train_graphs, test_graphs, train_labels, test_labels = \
                dataset_handler.split_dataset(graphs, labels)
            
            # Initialize classifier
            classifier = MelanomaClassifier(classifier_type=classifier_type)
            
            # Prepare features
            app.logger.info("Preparing features...")
            X_train = classifier.prepare_features(train_graphs)
            X_test = classifier.prepare_features(test_graphs)
            
            # Train and evaluate
            app.logger.info("Training and evaluating model...")
            results = classifier.train_evaluate(X_train, train_labels)
            
            # Save model
            model_path = os.path.join(MODEL_FOLDER, 'melanoma_classifier.joblib')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler.joblib')
            classifier.save_model(MODEL_FOLDER)
            
            # Get evaluation metrics
            accuracy = results['accuracy']
            precision = results['precision']
            recall = results['recall']
            f1 = results['f1']
            
            flash(f"Model trained successfully! Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}", "success")
            
            return render_template('train_results.html', 
                                  accuracy=accuracy,
                                  precision=precision,
                                  recall=recall,
                                  f1=f1,
                                  melanoma_count=len(melanoma_files),
                                  benign_count=len(benign_files))
            
        except Exception as e:
            app.logger.error(f"Error during training: {str(e)}")
            flash(f"Error during training: {str(e)}", "danger")
            return redirect(url_for('train'))
    
    return render_template('train.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded image and perform melanoma detection."""
    try:
        # Get the uploaded image
        if 'image' not in request.files:
            flash("No image selected", "danger")
            return redirect(url_for('index'))
        
        image_file = request.files['image']
        if image_file.filename == '':
            flash("No image selected", "danger")
            return redirect(url_for('index'))
        
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
        classifier = MelanomaClassifier(classifier_type='svm')
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
            
            # Check if model exists
            model_path = os.path.join(MODEL_FOLDER, 'melanoma_classifier.joblib')
            scaler_path = os.path.join(MODEL_FOLDER, 'scaler.joblib')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                flash("No trained model found. Please train the model first.", "warning")
                return redirect(url_for('train'))
            
            # Load model
            model = load(model_path)
            scaler = load(scaler_path)
            
            # Prepare features for prediction
            X = classifier.prepare_features([G])
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            prediction_label = "Melanoma" if prediction == 1 else "Benign"
            
            # Store result
            result = {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'original_filename': original_filename,
                'image_path': image_path,
                'superpixels_image_path': superpixels_image_path,
                'graph_image_path': graph_image_path,
                'prediction': prediction_label,
                'probability': probability,
                'features_extracted': len(X[0])
            }
            
            # store this in a database
            
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
        classifier = MelanomaClassifier(classifier_type='svm')
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

        # Load trained model
        model_path = os.path.join(MODEL_FOLDER, 'melanoma_classifier.joblib')
        scaler_path = os.path.join(MODEL_FOLDER, 'scaler.joblib')
        feature_selector_path = os.path.join(MODEL_FOLDER, 'feature_selector.joblib')

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
            X_scaled = feature_selector.transform(X_scaled)
            
        probability = model.predict_proba(X_scaled)[0][1]
        risk_level = "HIGH" if probability > 0.5 else "LOW"

        return jsonify({
            "status": 200,
            "message": "Image processed successfully",
            "data": {
                "probability": float(probability),
                "risk_level": risk_level,
                "validation": validation_message
            }}), 200

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": 500, "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)