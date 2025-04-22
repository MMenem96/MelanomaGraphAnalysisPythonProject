"""
CNN-based classifier for melanoma detection.
This module provides a CNN classifier using TensorFlow/Keras with options
for both training from scratch and using pretrained models.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CNNMelanomaClassifier:
    """CNN-based classifier for melanoma detection using TensorFlow/Keras."""
    
    def __init__(self, model_type='efficient_net', input_shape=(224, 224, 3)):
        """
        Initialize the CNN classifier.
        
        Args:
            model_type: Type of CNN architecture to use
                Options: 'custom', 'resnet50', 'efficient_net', 'inception_v3'
            input_shape: Input shape for the model (height, width, channels)
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Create model architecture
        self._build_model()
        
    def _build_model(self):
        """Build the CNN model architecture based on selected type."""
        try:
            if self.model_type == 'custom':
                self.logger.info("Building custom CNN model")
                self.model = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(128, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                
            elif self.model_type == 'resnet50':
                self.logger.info("Building ResNet50 model with transfer learning")
                base_model = applications.ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
                base_model.trainable = False  # Freeze base model
                
                inputs = tf.keras.Input(shape=self.input_shape)
                x = applications.resnet50.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(1024, activation='relu')(x)
                x = layers.Dropout(0.5)(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                self.model = tf.keras.Model(inputs, outputs)
                
            elif self.model_type == 'efficient_net':
                self.logger.info("Building EfficientNetB3 model with transfer learning")
                base_model = applications.EfficientNetB3(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
                base_model.trainable = False  # Freeze base model
                
                inputs = tf.keras.Input(shape=self.input_shape)
                x = applications.efficientnet.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dropout(0.5)(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                self.model = tf.keras.Model(inputs, outputs)
                
            elif self.model_type == 'inception_v3':
                self.logger.info("Building InceptionV3 model with transfer learning")
                base_model = applications.InceptionV3(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
                base_model.trainable = False  # Freeze base model
                
                inputs = tf.keras.Input(shape=self.input_shape)
                x = applications.inception_v3.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(1024, activation='relu')(x)
                x = layers.Dropout(0.5)(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                self.model = tf.keras.Model(inputs, outputs)
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            self.logger.info(f"Successfully built {self.model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
            
    def preprocess_images(self, image_paths, labels=None, target_size=None):
        """
        Preprocess images for CNN input.
        
        Args:
            image_paths: List of image file paths
            labels: Optional array of labels
            target_size: Optional tuple of (height, width) to resize images
            
        Returns:
            Preprocessed images as a numpy array and labels if provided
        """
        if target_size is None:
            target_size = self.input_shape[:2]
            
        try:
            # Load and preprocess images
            images = []
            for img_path in image_paths:
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    target_size=target_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                
            # Convert to numpy array and normalize
            images = np.array(images) / 255.0
            
            if labels is not None:
                return images, np.array(labels)
            return images
            
        except Exception as e:
            self.logger.error(f"Error preprocessing images: {str(e)}")
            raise
            
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, 
            unfreeze_layers=0, fine_tune_epochs=20, model_path='model/cnn_model.h5'):
        """
        Train the CNN model on the provided dataset.
        
        Args:
            X_train: Training images or paths to images
            y_train: Training labels
            X_val: Validation images or paths to images (if None, split from train)
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            unfreeze_layers: Number of layers to unfreeze for fine-tuning
            fine_tune_epochs: Number of epochs for fine-tuning
            model_path: Path to save best model
            
        Returns:
            Training history
        """
        try:
            # If X_train is a list of paths, load and preprocess images
            if isinstance(X_train[0], str):
                self.logger.info("Preprocessing training images from paths")
                if X_val is not None and y_val is not None:
                    X_train, y_train = self.preprocess_images(X_train, y_train)
                    X_val, y_val = self.preprocess_images(X_val, y_val)
                else:
                    X_train, y_train = self.preprocess_images(X_train, y_train)
                    # Split data for validation if not provided
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                    )
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    model_path,
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Train the model
            self.logger.info(f"Training {self.model_type} model for {epochs} epochs")
            self.history = self.model.fit(
                self.datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Fine-tuning phase if using transfer learning and unfreeze_layers > 0
            if self.model_type != 'custom' and unfreeze_layers > 0:
                self.logger.info(f"Fine-tuning last {unfreeze_layers} layers")
                
                # Unfreeze the last unfreeze_layers for fine-tuning
                if self.model_type == 'resnet50':
                    base_model = self.model.layers[2]  # Get the base model
                    for layer in base_model.layers[-unfreeze_layers:]:
                        layer.trainable = True
                        
                elif self.model_type in ['efficient_net', 'inception_v3']:
                    base_model = self.model.layers[2]  # Get the base model
                    for layer in base_model.layers[-unfreeze_layers:]:
                        layer.trainable = True
                
                # Recompile with a lower learning rate
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-5),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC()]
                )
                
                # Fine-tune with a slower learning rate
                fine_tune_history = self.model.fit(
                    self.datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    epochs=fine_tune_epochs,
                    callbacks=callbacks
                )
                
                # Combine histories
                for key in fine_tune_history.history:
                    self.history.history[key].extend(fine_tune_history.history[key])
            
            # Load the best weights
            self.model.load_weights(model_path)
            
            return self.history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X: Images or paths to images
            
        Returns:
            Predicted probabilities
        """
        try:
            # If X is a list of paths, load and preprocess images
            if isinstance(X[0], str):
                X = self.preprocess_images(X)
                
            # Get predictions
            return self.model.predict(X)
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test images or paths to images
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # If X_test is a list of paths, load and preprocess images
            if isinstance(X_test[0], str):
                X_test, y_test = self.preprocess_images(X_test, y_test)
                
            # Get predictions
            y_pred_prob = self.model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            
            # Compute sensitivity and specificity
            tn, fp, fn, tp = self._calculate_confusion_values(y_test, y_pred)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Log results
            self.logger.info(f"Evaluation results for {self.model_type}:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"AUC: {auc:.4f}")
            self.logger.info(f"Sensitivity: {sensitivity:.4f}")
            self.logger.info(f"Specificity: {specificity:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def _calculate_confusion_values(self, y_true, y_pred):
        """Calculate TP, FP, TN, FN from true and predicted labels."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn, fp, fn, tp
            
    def save_model(self, model_path='model/cnn_model.h5'):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, model_path='model/cnn_model.h5'):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            self.logger.warning("No training history available to plot")
            return
            
        try:
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'])
            ax1.plot(self.history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Validation'], loc='lower right')
            
            # Plot loss
            ax2.plot(self.history.history['loss'])
            ax2.plot(self.history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='upper right')
            
            # Tight layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"Training history plot saved to {save_path}")
                
            # Show plot
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")