import numpy as np
import logging
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from database import db


class AnalysisResult(db.Model):
    """Model for storing melanoma analysis results."""
    
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(String(255), unique=True, nullable=False)  # UUID for the analysis
    filename = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Prediction results
    prediction = Column(String(50), nullable=True)  # 'Melanoma' or 'Benign'
    probability = Column(Float, nullable=True)
    
    # Analysis metadata
    features_extracted = Column(Integer, nullable=False)
    n_segments = Column(Integer, nullable=False, default=20)
    compactness = Column(Float, nullable=False, default=10.0)
    connectivity_threshold = Column(Float, nullable=False, default=0.5)
    
    # Image paths
    original_image_path = Column(String(255), nullable=False)
    superpixels_image_path = Column(String(255), nullable=True)
    graph_image_path = Column(String(255), nullable=True)
    features_image_path = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<AnalysisResult {self.analysis_id} - {self.prediction}>"


class TrainingRecord(db.Model):
    """Model for storing model training records."""
    
    __tablename__ = 'training_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    classifier_type = Column(String(50), nullable=False)  # 'svm' or 'rf'
    
    # Training parameters
    n_segments = Column(Integer, nullable=False)
    compactness = Column(Float, nullable=False)
    connectivity_threshold = Column(Float, nullable=False)
    
    # Training data info
    melanoma_samples = Column(Integer, nullable=False)
    benign_samples = Column(Integer, nullable=False)
    
    # Evaluation metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    
    # Model path
    model_path = Column(String(255), nullable=False)
    
    def __repr__(self):
        return f"<TrainingRecord {self.id} - {self.classifier_type}>"


class MelanomaClassifier:
    """
    Classifier for melanoma detection using extracted features.
    Supports SVM and Random Forest classifiers.
    """
    
    def __init__(self, classifier_type='svm'):
        """
        Initialize the classifier.
        
        Args:
            classifier_type: Type of classifier ('svm' or 'rf')
        """
        self.logger = logging.getLogger(__name__)
        self.classifier_type = classifier_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize the model based on type
        if classifier_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            self.logger.info("Initialized SVM classifier with RBF kernel")
        elif classifier_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.logger.info("Initialized Random Forest classifier with 100 trees")
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def train(self, X, y):
        """
        Train the classifier on extracted features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0: benign, 1: melanoma)
        """
        try:
            # Validate input dimensions
            if len(X) != len(y):
                raise ValueError(f"X and y dimensions don't match: {len(X)} vs {len(y)}")
            
            # Check if there are enough samples for each class
            unique_labels, counts = np.unique(y, return_counts=True)
            self.logger.info(f"Training data distribution: {dict(zip(unique_labels, counts))}")
            
            if len(unique_labels) < 2:
                raise ValueError("Need examples from both classes for training")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the model
            self.model.fit(X_scaled, y)
            self.logger.info(f"Trained {self.classifier_type} classifier on {len(X)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels and probabilities
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: True labels
            
        Returns:
            Accuracy, precision, recall, F1-score, AUC, and confusion matrix
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_prob)
            cm = confusion_matrix(y, y_pred)
            
            self.logger.info(f"Evaluation results:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"AUC: {auc:.4f}")
            
            return accuracy, precision, recall, f1, auc, cm
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def save(self, model_path):
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No trained model to save")
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and scaler as a tuple
            joblib.dump((self.model, self.scaler), model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self, model_path):
        """
        Load a trained model and scaler.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load model and scaler
            loaded_data = joblib.load(model_path)
            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                self.model, self.scaler = loaded_data
            else:
                self.logger.error("Invalid model format")
                return False
            
            # Validate the loaded model
            if not hasattr(self.model, 'predict_proba'):
                self.logger.error("Loaded model does not have predict_proba method")
                return False
            
            # Determine classifier type from loaded model
            if isinstance(self.model, SVC):
                self.classifier_type = 'svm'
            elif isinstance(self.model, RandomForestClassifier):
                self.classifier_type = 'rf'
            else:
                self.classifier_type = 'unknown'
            
            self.logger.info(f"Loaded {self.classifier_type} model from {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False