"""
CNN-based classifier for BCC vs SK detection.
This module provides a CNN classifier using TensorFlow/Keras with options
for both training from scratch and using pretrained models.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from keras import layers, models, applications
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CNNBCCSKClassifier:
    """CNN-based classifier for BCC vs SK detection using TensorFlow/Keras."""
    
    def __init__(self, model_type='efficient_net', input_shape=(224, 224, 3), enhanced=False):
        """
        Initialize the CNN classifier.
        
        Args:
            model_type: Type of CNN architecture to use
                Options: 'custom', 'resnet50', 'efficient_net', 'inception_v3', 'enhanced_efficientnet'
            input_shape: Input shape for the model (height, width, channels)
            enhanced: Whether to use enhanced data augmentation and training strategies
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.enhanced = enhanced
        
        # Track feature counts for model summary
        self.feature_counts = {
            'total': 0,
            'used': 0,
            'frozen': 0
        }
        
        # Track hyperparameters used in training
        self.hyperparameters = {
            'model_type': model_type,
            'input_shape': input_shape,
            'enhanced': enhanced,
            'mixup_alpha': 0.2,  # Default value, will be updated during training
            'fine_tune_epochs': 0,
            'unfreeze_layers': 0,
            'total_epochs': 0,
            'batch_size': 32
        }
        
        # Enhanced data augmentation for skin lesions
        if enhanced:
            self.datagen = ImageDataGenerator(
                rotation_range=30,           # More rotation for skin lesions
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.3,              # More zoom variation
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                brightness_range=[0.7, 1.3], # Brightness variation
                channel_shift_range=20.0,    # Color variation
                # Advanced color augmentation
                preprocessing_function=self._color_augmentation
            )
        else:
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
        
    def _color_augmentation(self, img):
        """Advanced color augmentation for dermoscopic images"""
        # Convert to HSV for better color manipulation
        if np.random.random() > 0.5:
            # Apply only 50% of the time
            return img
            
        try:
            from skimage import color
            img = img / 255.0  # Normalize to 0-1
            
            # Random contrast adjustment
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)
            img = alpha * img
            
            # Random gamma adjustment (simulates different lighting)
            gamma = np.random.uniform(0.8, 1.2)
            img = np.power(img, gamma)
            
            # Clip to valid range
            img = np.clip(img, 0, 1)
            
            # Convert back to 0-255 range
            img = img * 255.0
            
            return img.astype(np.uint8)
        except:
            # If anything fails, return original image
            self.logger.warning("Color augmentation failed, returning original image")
            return img
        
    def _count_model_parameters(self):
        """Count and log the model parameters."""
        if self.model is None:
            return
            
        # Track model parameters
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        self.feature_counts['total'] = trainable_count + non_trainable_count
        self.feature_counts['used'] = trainable_count
        self.feature_counts['frozen'] = non_trainable_count
        self.logger.info(f"Model has {self.feature_counts['total']:,} total parameters, {self.feature_counts['used']:,} trainable")
        
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
                
            elif self.model_type == 'enhanced_efficientnet':
                self.logger.info("Building Enhanced EfficientNetB3 model with optimized transfer learning")
                # Use EfficientNetB3 as base with imagenet weights
                base_model = applications.EfficientNetB3(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
                
                # Freeze early layers but allow later layers to be trained
                for layer in base_model.layers:
                    layer.trainable = False
                
                # Unfreeze the top layers (last 30% of layers)
                trainable_layers = int(len(base_model.layers) * 0.3)
                for layer in base_model.layers[-trainable_layers:]:
                    layer.trainable = True
                
                self.logger.info(f"Made last {trainable_layers} layers of EfficientNetB3 trainable")
                
                # Build model with more sophisticated architecture
                inputs = tf.keras.Input(shape=self.input_shape)
                x = applications.efficientnet.preprocess_input(inputs)
                
                # Base model with fine-tuning
                x = base_model(x)
                
                # Advanced pooling (combines average and max pooling)
                avg_pool = layers.GlobalAveragePooling2D()(x)
                max_pool = layers.GlobalMaxPooling2D()(x)
                x = layers.Concatenate()([avg_pool, max_pool])
                
                # Batch normalization for faster convergence and stability
                x = layers.BatchNormalization()(x)
                
                # First dense block with residual connection
                skip = x
                x = layers.Dense(512, activation=None)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.Dropout(0.4)(x)
                x = layers.Dense(512, activation=None)(x)
                x = layers.BatchNormalization()(x)
                # Add skip connection (needs to match dimensions)
                skip = layers.Dense(512)(skip)
                x = layers.Add()([x, skip])
                x = layers.Activation('relu')(x)
                x = layers.Dropout(0.4)(x)
                
                # Final classification layers
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
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
                
            # Count model parameters
            self._count_model_parameters()
            
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
            unfreeze_layers=0, fine_tune_epochs=20, model_path='model/bcc_sk_cnn_model.h5', mixup_alpha=0.2):
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
            mixup_alpha: Alpha parameter for MixUp data augmentation (0.2-0.4 recommended for skin lesions)
            
        Returns:
            Training history
        """
        # Store hyperparameters for later reference
        self.hyperparameters.update({
            'total_epochs': epochs + fine_tune_epochs,
            'mixup_alpha': mixup_alpha,
            'batch_size': batch_size,
            'fine_tune_epochs': fine_tune_epochs,
            'unfreeze_layers': unfreeze_layers
        })
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
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Enhanced callbacks and training strategy for the enhanced model
            if self.enhanced or self.model_type == 'enhanced_efficientnet':
                self.logger.info("Using enhanced training strategy with stronger data augmentation")
                
                # Enhanced callbacks for better training
                callbacks = [
                    # Early stopping with more patience for complex models
                    EarlyStopping(
                        monitor='val_auc',
                        patience=15,
                        restore_best_weights=True,
                        mode='max',
                        verbose=1
                    ),
                    # Checkpoint that saves the best model based on AUC
                    ModelCheckpoint(
                        model_path,
                        monitor='val_auc',  # Use AUC for medical image classification
                        save_best_only=True,
                        mode='max',
                        verbose=1
                    ),
                    # Gradual learning rate reduction
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,      # More gradual reduction
                        patience=8,
                        min_lr=1e-7,
                        verbose=1
                    )
                ]
                
                # Mixup data augmentation for skin lesion classification
                from tensorflow.keras.utils import Sequence
                class MixupGenerator(Sequence):
                    def __init__(self, x, y, batch_size=32, alpha=0.2):
                        self.x = x
                        self.y = y
                        self.batch_size = batch_size
                        self.alpha = alpha
                        self.indices = np.arange(len(x))
                        np.random.shuffle(self.indices)
                        
                    def __len__(self):
                        return int(np.ceil(len(self.x) / self.batch_size))
                        
                    def __getitem__(self, idx):
                        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_x = self.x[batch_indices]
                        batch_y = self.y[batch_indices]
                        
                        # Apply standard augmentations using the datagen
                        for i in range(len(batch_x)):
                            batch_x[i] = self.datagen.random_transform(batch_x[i])
                        
                        # Apply mixup to half the batches
                        if np.random.random() > 0.5:
                            # Create mixup batch
                            batch_indices_2 = np.random.choice(self.indices, len(batch_indices))
                            batch_x_2 = self.x[batch_indices_2]
                            batch_y_2 = self.y[batch_indices_2]
                            
                            # Apply standard augmentations to second batch
                            for i in range(len(batch_x_2)):
                                batch_x_2[i] = self.datagen.random_transform(batch_x_2[i])
                            
                            # Generate random lambdas
                            l = np.random.beta(self.alpha, self.alpha, len(batch_indices))
                            l = np.reshape(l, (len(batch_indices), 1, 1, 1))
                            l_y = np.reshape(l, (len(batch_indices), 1))
                            
                            # Create mixed batches
                            batch_x = batch_x * l + batch_x_2 * (1 - l)
                            batch_y = batch_y * l_y + batch_y_2 * (1 - l_y)
                        
                        return batch_x, batch_y
                        
                    def on_epoch_end(self):
                        np.random.shuffle(self.indices)
                
                # Use enhanced training for the model
                self.logger.info(f"Training {self.model_type} model with mixup augmentation for {epochs} epochs")
                
                # Modified optimizer for enhanced training
                # Cyclic learning rate often works better for CNN training
                initial_learning_rate = 1e-4
                maximal_learning_rate = 1e-3
                step_size = 8 * (len(X_train) // batch_size)  # 8 epochs worth of steps
                
                lr_schedule = tf.keras.optimizers.schedules.CyclicLR(
                    initial_learning_rate=initial_learning_rate,
                    maximal_learning_rate=maximal_learning_rate,
                    step_size=step_size,
                    scale_fn=lambda x: 1/(2.**(x-1)),  # Exponential decay
                    scale_mode='cycle'
                )
                
                # Recompile with cyclic learning rate
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC()]
                )
                
                # Train with mixup (using the provided alpha parameter)
                self.logger.info(f"Creating MixUp generator with alpha={mixup_alpha}")
                mixup_gen = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=mixup_alpha)
                mixup_gen.datagen = self.datagen  # Share datagen for consistency
                
                self.history = self.model.fit(
                    mixup_gen,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
                
            else:
                # Standard callbacks for regular models
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
                
                # Train the model - standard approach
                self.logger.info(f"Training {self.model_type} model for {epochs} epochs")
                self.history = self.model.fit(
                    self.datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    callbacks=callbacks
                )
            
            # Fine-tuning phase if using transfer learning and unfreeze_layers > 0
            # Skip for enhanced_efficientnet since it already has fine-tuning built in
            if self.model_type != 'custom' and self.model_type != 'enhanced_efficientnet' and unfreeze_layers > 0:
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
                
                # Update parameter counts after unfreezing layers
                self._count_model_parameters()
                
                # Fine-tune with a slower learning rate
                # Use the same enhanced training approach if this is an enhanced model
                if self.enhanced:
                    self.logger.info("Using enhanced fine-tuning strategy")
                    # Create mixup generator for fine-tuning phase with same alpha parameter
                    self.logger.info(f"Creating MixUp generator for fine-tuning with alpha={mixup_alpha}")
                    mixup_gen = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=mixup_alpha)
                    mixup_gen.datagen = self.datagen
                    
                    fine_tune_history = self.model.fit(
                        mixup_gen,
                        validation_data=(X_val, y_val),
                        epochs=fine_tune_epochs,
                        callbacks=callbacks,
                        verbose=1
                    )
                else:
                    # Standard fine-tuning
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
            
            # Get model summary
            model_summary = self.get_model_summary()
            
            # Add evaluation metrics to the results
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
            
            # Add model summary information
            results['model_info'] = model_summary
            
            return results
            
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
            
    def save_model(self, model_path='model/bcc_sk_cnn_model.h5'):
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
            
    def load_model(self, model_path='model/bcc_sk_cnn_model.h5'):
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
            
    def get_model_summary(self):
        """
        Get a dictionary summary of model architecture, hyperparameters, and feature counts.
        
        Returns:
            Dictionary containing model summary information
        """
        return {
            'model_type': self.model_type,
            'enhanced': self.enhanced,
            'input_shape': self.input_shape,
            'hyperparameters': self.hyperparameters,
            'feature_counts': self.feature_counts
        }
        
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