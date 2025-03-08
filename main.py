from flask import Flask, request, jsonify
import os
from preprocessing import ImagePreprocessor
from superpixel import SuperpixelGenerator
from graph_construction import GraphConstructor
from feature_extraction import FeatureExtractor
from classifier import MelanomaClassifier
from joblib import load

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
        image_path = "uploads/temp.jpg"
        image_file.save(image_path)

        # Process Image
        preprocessor = ImagePreprocessor()
        superpixel_gen = SuperpixelGenerator(n_segments=20, compactness=10)
        graph_constructor = GraphConstructor(connectivity_threshold=0.5)
        feature_extractor = FeatureExtractor()
        classifier = MelanomaClassifier(classifier_type='svm')

        image = preprocessor.load_image(image_path)
        processed_image = preprocessor.preprocess(image)

        # Generate superpixels
        segments = superpixel_gen.generate_superpixels(processed_image)
        features = superpixel_gen.compute_superpixel_features(processed_image, segments)

        # Construct graph
        G = graph_constructor.build_graph(features, segments)

        # Extract features
        G.graph['features'] = {
            **feature_extractor.extract_local_features(G),
            **feature_extractor.extract_global_features(G),
            **feature_extractor.extract_spectral_features(G)
        }

        # Load trained model
        model_path = 'model/melanoma_classifier.joblib'
        scaler_path = 'model/scaler.joblib'

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": "Model not found. Train the model first."}), 500

        model = load(model_path)
        scaler = load(scaler_path)

        # Prepare features for prediction
        X = classifier.prepare_features([G])
        X_scaled = scaler.transform(X)
        probability = model.predict_proba(X_scaled)[0][1]
        risk_level = "HIGH" if probability > 0.5 else "LOW"

        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "data": {
                "probability": probability,
                "risk_level": risk_level
            }},200
    )

    except Exception as e:
        return jsonify({"status": 500,"message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
