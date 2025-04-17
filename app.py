from flask import Flask, request, jsonify
import os
from src.preprocessing import ImagePreprocessor
from src.superpixel import SuperpixelGenerator
from src.graph_construction import GraphConstructor
from src.feature_extraction import FeatureExtractor
from src.conventional_features import ConventionalFeatureExtractor
from src.classifier import MelanomaClassifier
from src.image_validator import ImageValidator
from joblib import load
import cv2
import numpy as np
import logging

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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
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
        model_path = 'model/melanoma_classifier.joblib'
        scaler_path = 'model/scaler.joblib'
        feature_selector_path = 'model/feature_selector.joblib'

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