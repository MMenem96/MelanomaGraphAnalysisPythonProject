"""
Modern Deep Neural Network Classifier for Tabular/Handcrafted Features

This module implements state-of-the-art deep learning architectures optimized for
tabular data classification, incorporating:
- Deep residual networks with skip connections
- Batch normalization and dropout regularization
- Self-attention mechanisms for feature importance
- Advanced training techniques (focal loss, mixup augmentation)
- Optimized for medical image feature classification

Author: Melanoma Graph Analysis Project
Date: January 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.class_weight import compute_class_weight
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import os

warnings.filterwarnings('ignore', category=UserWarning)


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard misclassified examples. It's particularly effective for
    highly imbalanced datasets.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        alpha: Weighting factor for minority class (0-1)
        gamma: Focusing parameter for modulating loss (default: 2.0)
        from_logits: Whether input is logits or probabilities
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """Compute focal loss"""
        epsilon = tf.keras.backend.epsilon()
        
        # Ensure predictions are probabilities
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute focal loss
        y_true = tf.cast(y_true, tf.float32)
        
        # Binary focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        
        loss = -focal_weight * tf.math.log(pt)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class MixupGenerator(Sequence):
    """
    Data generator with Mixup augmentation for tabular data.
    
    Mixup creates virtual training examples by linearly interpolating
    between random pairs of samples and their labels.
    
    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)
    
    Args:
        X: Feature array
        y: Label array
        batch_size: Number of samples per batch
        alpha: Mixup interpolation parameter (default: 0.2)
        shuffle: Whether to shuffle data each epoch
    """
    
    def __init__(self, X, y, batch_size=32, alpha=0.2, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data with mixup"""
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        # Apply mixup
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, size=(len(X_batch), 1))
            
            # Random permutation for mixing
            perm_indexes = np.random.permutation(len(X_batch))
            X_batch_permuted = X_batch[perm_indexes]
            y_batch_permuted = y_batch[perm_indexes]
            
            # Mix samples
            X_batch = lam * X_batch + (1 - lam) * X_batch_permuted
            y_batch = lam * y_batch + (1 - lam) * y_batch_permuted
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)


class TabularDNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Modern Deep Neural Network for tabular/handcrafted feature classification.
    
    This classifier implements a deep residual architecture optimized for
    tabular data with medical imaging features. It incorporates:
    - Multiple residual blocks with skip connections
    - Batch normalization for stable training
    - Dropout regularization to prevent overfitting
    - Self-attention mechanism for feature importance
    - Advanced training with focal loss and mixup augmentation
    
    Compatible with scikit-learn API for easy integration.
    
    Args:
        input_dim: Number of input features (default: 511)
        hidden_units: List of hidden layer sizes (default: [512, 512, 384, 384, 256, 256, 128])
        dropout_rate: Dropout rate for regularization (default: 0.3)
        l2_reg: L2 regularization coefficient (default: 1e-4)
        activation: Activation function ('swish', 'relu', 'elu') (default: 'swish')
        use_attention: Whether to use attention mechanism (default: True)
        learning_rate: Initial learning rate (default: 1e-3)
        batch_size: Batch size for training (default: 128)
        epochs: Maximum number of training epochs (default: 200)
        patience: Early stopping patience (default: 30)
        focal_loss_alpha: Focal loss alpha parameter for class weight (default: 0.16)
        focal_loss_gamma: Focal loss gamma parameter for focusing (default: 2.0)
        mixup_alpha: Mixup augmentation alpha (0=disabled) (default: 0.2)
        validation_split: Fraction of data for validation (default: 0.2)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed) (default: 1)
        random_state: Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, 
                 input_dim=511,
                 hidden_units=None,
                 dropout_rate=0.3,
                 l2_reg=1e-4,
                 activation='swish',
                 use_attention=True,
                 learning_rate=1e-3,
                 batch_size=128,
                 epochs=200,
                 patience=30,
                 focal_loss_alpha=0.16,
                 focal_loss_gamma=2.0,
                 mixup_alpha=0.2,
                 validation_split=0.2,
                 verbose=1,
                 random_state=42):
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units if hidden_units is not None else [512, 512, 384, 384, 256, 256, 128]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.activation = activation
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.mixup_alpha = mixup_alpha
        self.validation_split = validation_split
        self.verbose = verbose
        self.random_state = random_state
        
        self.model_ = None
        self.history_ = None
        self.classes_ = None
        self.n_classes_ = None
        
        # Set random seeds
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def _build_residual_block(self, x, units, dropout_rate, block_name):
        """
        Build a residual block with skip connections.
        
        Args:
            x: Input tensor
            units: Number of units in the block
            dropout_rate: Dropout rate
            block_name: Name prefix for the block layers
        
        Returns:
            Output tensor after residual block
        """
        shortcut = x
        
        # First dense layer
        x = layers.Dense(
            units, 
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name=f'{block_name}_dense1'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Activation(self.activation, name=f'{block_name}_act1')(x)
        x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
        
        # Second dense layer
        x = layers.Dense(
            units, 
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name=f'{block_name}_dense2'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        
        # Projection shortcut if dimensions don't match
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units, name=f'{block_name}_projection')(shortcut)
        
        # Add skip connection
        x = layers.Add(name=f'{block_name}_add')([x, shortcut])
        x = layers.Activation(self.activation, name=f'{block_name}_act2')(x)
        x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout2')(x)
        
        return x
    
    def _build_model(self):
        """Build the deep residual neural network model"""
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # Embedding layer - expand feature space
        x = layers.Dense(
            self.hidden_units[0], 
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='embedding_dense'
        )(inputs)
        x = layers.BatchNormalization(name='embedding_bn')(x)
        x = layers.Activation(self.activation, name='embedding_activation')(x)
        x = layers.Dropout(self.dropout_rate, name='embedding_dropout')(x)
        
        # Stack of residual blocks
        prev_units = self.hidden_units[0]
        block_idx = 0
        
        for i, units in enumerate(self.hidden_units):
            # Gradually decrease dropout rate in deeper layers
            layer_dropout = self.dropout_rate * (1 - 0.3 * i / len(self.hidden_units))
            
            x = self._build_residual_block(
                x, 
                units, 
                layer_dropout, 
                block_name=f'residual_block_{block_idx}'
            )
            block_idx += 1
            
            # Add second residual block at same dimension for deeper layers
            if i < len(self.hidden_units) - 1:
                x = self._build_residual_block(
                    x, 
                    units, 
                    layer_dropout, 
                    block_name=f'residual_block_{block_idx}'
                )
                block_idx += 1
            
            prev_units = units
        
        # Self-attention mechanism for feature importance
        if self.use_attention:
            attention = layers.Dense(prev_units, activation='tanh', name='attention_tanh')(x)
            attention = layers.Dense(prev_units, activation='softmax', name='attention_softmax')(attention)
            x = layers.Multiply(name='attention_multiply')([x, attention])
        
        # Final dense layer before output
        x = layers.Dense(
            64, 
            activation=self.activation,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='pre_output_dense'
        )(x)
        x = layers.BatchNormalization(name='pre_output_bn')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='pre_output_dropout')(x)
        
        # Output layer (sigmoid for binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Build model
        model = models.Model(inputs=inputs, outputs=outputs, name='TabularDNN')
        
        return model
    
    def _get_callbacks(self):
        """Create training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=self.patience,
                restore_best_weights=True,
                mode='max',
                verbose=self.verbose
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=max(10, self.patience // 3),
                min_lr=1e-7,
                mode='max',
                verbose=self.verbose
            )
        ]
        
        return callbacks_list
    
    def fit(self, X, y):
        """
        Train the deep neural network classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        
        Returns:
            self: Trained classifier
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float32)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ != 2:
            raise ValueError(f"TabularDNNClassifier only supports binary classification. "
                           f"Found {self.n_classes_} classes.")
        
        # Update input dimension from actual data
        self.input_dim = X.shape[1]
        
        # Build model
        self.model_ = self._build_model()
        
        # Compile model with AdamW optimizer and focal loss
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model_.compile(
            optimizer=optimizer,
            loss=FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        if self.verbose >= 2:
            self.model_.summary()
        
        # Prepare training data
        y_train = y.reshape(-1, 1).astype(np.float32)
        
        # Split validation data
        if self.validation_split > 0:
            n_val = int(len(X) * self.validation_split)
            val_indices = np.random.choice(len(X), n_val, replace=False)
            train_indices = np.array([i for i in range(len(X)) if i not in val_indices])
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train_split, y_val = y_train[train_indices], y_train[val_indices]
            
            validation_data = (X_val, y_val)
        else:
            X_train = X
            y_train_split = y_train
            validation_data = None
        
        # Train with or without mixup
        if self.mixup_alpha > 0 and validation_data is not None:
            # Use mixup generator for training
            train_generator = MixupGenerator(
                X_train, 
                y_train_split, 
                batch_size=self.batch_size,
                alpha=self.mixup_alpha,
                shuffle=True
            )
            
            self.history_ = self.model_.fit(
                train_generator,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=self._get_callbacks(),
                verbose=self.verbose
            )
        else:
            # Standard training without mixup
            self.history_ = self.model_.fit(
                X_train,
                y_train_split,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split if validation_data is None else 0,
                validation_data=validation_data,
                callbacks=self._get_callbacks(),
                verbose=self.verbose
            )
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Probabilities for each class, shape (n_samples, n_classes)
        """
        check_is_fitted(self, ['model_', 'classes_'])
        X = check_array(X, dtype=np.float32)
        
        # Get probability for class 1
        proba_class_1 = self.model_.predict(X, batch_size=self.batch_size, verbose=0).flatten()
        
        # Return probabilities for both classes
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted class labels, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        return self.classes_[predictions]
    
    def score(self, X, y):
        """
        Return accuracy score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'activation': self.activation,
            'use_attention': self.use_attention,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'focal_loss_alpha': self.focal_loss_alpha,
            'focal_loss_gamma': self.focal_loss_gamma,
            'mixup_alpha': self.mixup_alpha,
            'validation_split': self.validation_split,
            'verbose': self.verbose,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def visualize_architecture(self, save_path=None, dpi=300, style='neurons'):
        """
        Create publication-quality visualization of the DNN architecture.
        
        Args:
            save_path (str, optional): Path to save the figure. Auto-generated if None.
            dpi (int): Resolution for publication quality (default: 300)
            style (str): 'neurons' for traditional neuron diagram, 'blocks' for architecture blocks
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if style == 'neurons':
            return self._visualize_neurons(save_path, dpi)
        else:
            return self._visualize_blocks(save_path, dpi)
    
    def _visualize_neurons(self, save_path=None, dpi=300):
        """
        Create traditional neuron-based network diagram like academic papers.
        Shows individual neurons, connections, weight matrices, and activation functions.
        """
        try:
            fig, ax = plt.subplots(figsize=(18, 7))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Define vibrant but professional colors for layers (scientific palette)
            layer_colors = [
                '#FFE5CC',  # Peach - Input
                '#CCE5FF',  # Light Blue - Hidden 1
                '#FFFFCC',  # Light Yellow - Hidden 2
                '#CCFFDD',  # Light Green - Hidden 3
                '#FFCCFF',  # Light Pink - Hidden 4
                '#FFCCCC',  # Light Red - Hidden 5
                '#CCFFFF',  # Cyan - Hidden 6
                '#FFE5DD',  # Light Peach - Hidden 7
                '#E5E5E5'   # Light Gray - Output
            ]
            
            # Calculate layer positions
            layers = [self.input_dim] + self.hidden_units + [1]  # Include input and output
            n_layers = len(layers)
            
            # Horizontal spacing
            x_positions = np.linspace(0.7, 9.3, n_layers)
            
            # Maximum neurons to display per layer (for visualization clarity)
            max_display_neurons = 5
            
            # Store neuron positions for drawing connections
            neuron_positions = []
            
            # ========== Draw each layer ==========
            for layer_idx, (n_neurons, x_pos) in enumerate(zip(layers, x_positions)):
                # Determine how many neurons to actually draw
                n_display = min(n_neurons, max_display_neurons)
                show_ellipsis = n_neurons > max_display_neurons
                
                # Calculate vertical positions
                y_center = 5.0
                if n_display == 1:
                    y_positions = [y_center]
                else:
                    y_spacing = min(4.0 / n_display, 0.95)
                    y_start = y_center - (n_display - 1) * y_spacing / 2
                    y_positions = [y_start + i * y_spacing for i in range(n_display)]
                
                layer_neurons = []
                
                # Draw layer background first
                if n_display >= 1:
                    y_min = min(y_positions) - 0.55 if n_display > 1 else y_positions[0] - 0.55
                    y_max = max(y_positions) + 0.55 if n_display > 1 else y_positions[0] + 0.55
                    width = 0.65
                    rect = FancyBboxPatch(
                        (x_pos - width/2, y_min), width, y_max - y_min,
                        boxstyle="round,pad=0.05",
                        facecolor=layer_colors[layer_idx % len(layer_colors)],
                        edgecolor='#333',
                        linewidth=1.5,
                        alpha=0.7,
                        zorder=1
                    )
                    ax.add_patch(rect)
                
                # Draw neurons
                for i, y_pos in enumerate(y_positions):
                    # Draw circle for neuron
                    circle = plt.Circle((x_pos, y_pos), 0.18, 
                                       facecolor='white', 
                                       edgecolor='black', 
                                       linewidth=2.2, 
                                       zorder=3)
                    ax.add_patch(circle)
                    layer_neurons.append((x_pos, y_pos))
                
                # Draw ellipsis if needed
                if show_ellipsis:
                    ellipsis_y = y_positions[-1] - 0.65
                    ax.text(x_pos, ellipsis_y, '⋮', 
                           ha='center', va='center', fontsize=20, fontweight='bold', zorder=2, color='#000')
                
                neuron_positions.append(layer_neurons)
                
                # ========== Draw connections to previous layer ==========
                if layer_idx > 0:
                    prev_neurons = neuron_positions[layer_idx - 1]
                    curr_neurons = neuron_positions[layer_idx]
                    
                    # Draw professional connections - showing fully connected pattern
                    # Draw all connections but with varying alpha based on position
                    for i, (x1, y1) in enumerate(prev_neurons):
                        for j, (x2, y2) in enumerate(curr_neurons):
                            # Emphasize edge connections, make middle ones lighter
                            if (i == 0 and j == 0) or (i == len(prev_neurons)-1 and j == len(curr_neurons)-1):
                                # Edge to edge - most prominent
                                alpha = 0.5
                                linewidth = 1.2
                            elif i in [0, len(prev_neurons)-1] or j in [0, len(curr_neurons)-1]:
                                # Connections involving edge neurons
                                alpha = 0.3
                                linewidth = 0.9
                            else:
                                # Middle connections - lighter
                                alpha = 0.15
                                linewidth = 0.7
                            
                            ax.plot([x1 + 0.18, x2 - 0.18], [y1, y2], 
                                   'k-', linewidth=linewidth, alpha=alpha, zorder=0)
                    
                    # Draw weight matrix label
                    x_mid = (x_positions[layer_idx-1] + x_pos) / 2
                    y_top = 7.5
                    ax.text(x_mid, y_top, f'$W_{{{layer_idx}}}$', 
                           ha='center', va='center', fontsize=14, 
                           style='italic', fontweight='bold', color='#000')
                    
                    # Add activation function label (except for last layer)
                    if layer_idx < n_layers - 1:
                        activation_label = self.activation.upper() if hasattr(self, 'activation') else 'SWISH'
                        ax.text(x_pos, 1.8, activation_label, 
                               ha='center', va='center', fontsize=14, 
                               style='italic', color='#000', fontweight='bold')
            
            # ========== Add layer labels ==========
            # Input layer
            ax.text(x_positions[0], 8.3, 'Input', ha='center', fontsize=14, fontweight='bold')
            ax.text(x_positions[0], 7.95, f'{self.input_dim}', ha='center', fontsize=14, color='#555')
            
            # Hidden layers
            for i in range(1, n_layers - 1):
                ax.text(x_positions[i], 8.3, f'Hidden {i}', ha='center', fontsize=14, fontweight='bold')
                ax.text(x_positions[i], 7.95, f'{layers[i]} neurons', ha='center', fontsize=14, color='#555')
            
            # Output layer with sigmoid activation
            ax.text(x_positions[-1], 8.3, 'Output', ha='center', fontsize=14, fontweight='bold')
            ax.text(x_positions[-1], 7.95, 'BCC/BKL', ha='center', fontsize=14, color='#555')
            ax.text(x_positions[-1], 1.8, 'Sigmoid', 
                   ha='center', va='center', fontsize=14, style='italic', color='#000', fontweight='bold')
            
            # ========== Title ==========
            title = 'Deep Neural Network Architecture'
            subtitle = f'{self.input_dim} Input Features → {len(self.hidden_units)} Hidden Layers → Binary Classification'
            ax.text(5, 9.4, title, ha='center', fontsize=14, fontweight='bold')
            ax.text(5, 9.0, subtitle, ha='center', fontsize=14, color='#444')
            
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                output_dir = Path('output/architecture')
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / 'dnn_neuron_diagram.png'
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=True)
            print(f"✅ Neuron-based DNN diagram saved to: {save_path}")
            
            plt.close(fig)
            return fig
            
        except Exception as e:
            print(f"❌ Error creating neuron visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_blocks(self, save_path=None, dpi=300):
        """Original block-based architecture visualization (kept for reference)"""
        try:
            # Create figure
            if style == 'detailed':
                fig, ax = plt.subplots(figsize=(18, 10))
            else:
                fig, ax = plt.subplots(figsize=(14, 8))
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Define colors for different layer types
            colors = {
                'input': '#ff7f0e',      # Orange
                'dense': '#1f77b4',      # Blue
                'residual': '#2ca02c',   # Green
                'attention': '#d62728',  # Red
                'dropout': '#9467bd',    # Purple
                'batchnorm': '#8c564b',  # Brown
                'output': '#2ca02c'      # Green
            }
            
            # Calculate layer positions
            n_blocks = sum([2 if i < len(self.hidden_units) - 1 else 1 for i in range(len(self.hidden_units))])
            total_sections = 2 + n_blocks + (1 if self.use_attention else 0) + 1  # input + embedding + residual blocks + attention + pre-output + output
            
            x_start = 0.8
            x_end = 9.2
            x_spacing = (x_end - x_start) / total_sections
            y_center = 5
            
            current_x = x_start
            layer_info = []
            
            # ========== Input Layer ==========
            self._draw_layer_node(
                ax, current_x, y_center, 
                f'Input\\n{self.input_dim}\\nfeatures',
                colors['input'], 0.6, 1.0, size=14
            )
            layer_info.append((current_x, y_center))
            current_x += x_spacing
            
            # ========== Embedding Layer ==========
            self._draw_layer_node(
                ax, current_x, y_center,
                f'Dense {self.hidden_units[0]}\\nBN+{self.activation[:4]}\\nDrop({self.dropout_rate:.1f})',
                colors['dense'], 0.6, 1.0, size=14
            )
            self._draw_arrow(ax, layer_info[-1][0] + 0.3, layer_info[-1][1], 
                           current_x - 0.3, y_center)
            layer_info.append((current_x, y_center))
            current_x += x_spacing * 1.2
            
            # ========== Residual Blocks ==========
            for i, units in enumerate(self.hidden_units):
                layer_dropout = self.dropout_rate * (1 - 0.3 * i / len(self.hidden_units))
                
                # Draw residual block with skip connection
                self._draw_residual_block_compact(
                    ax, current_x, y_center, units, layer_dropout, 
                    layer_info[-1], colors, i
                )
                
                layer_info.append((current_x, y_center))
                current_x += x_spacing * 1.2
                
                # Second block for non-final layers
                if i < len(self.hidden_units) - 1:
                    self._draw_residual_block_compact(
                        ax, current_x, y_center, units, layer_dropout,
                        layer_info[-1], colors, f'{i}b'
                    )
                    layer_info.append((current_x, y_center))
                    current_x += x_spacing * 1.2
            
            # ========== Attention Mechanism ==========
            if self.use_attention:
                # Draw attention box
                self._draw_layer_node(
                    ax, current_x, y_center + 1.2,
                    f'Attention\\nWeights',
                    colors['attention'], 0.5, 0.7, size=9
                )
                self._draw_arrow(ax, layer_info[-1][0] + 0.3, layer_info[-1][1],
                               current_x - 0.25, y_center + 1.2, style='dashed')
                
                # Multiply node
                self._draw_layer_node(
                    ax, current_x, y_center,
                    '×',
                    colors['attention'], 0.4, 0.4, size=14
                )
                self._draw_arrow(ax, current_x, y_center + 0.85, current_x, y_center + 0.2)
                self._draw_arrow(ax, layer_info[-1][0] + 0.3, layer_info[-1][1],
                               current_x - 0.2, y_center)
                
                layer_info.append((current_x, y_center))
                current_x += x_spacing
            
            # ========== Pre-output ==========
            self._draw_layer_node(
                ax, current_x, y_center,
                f'Dense 64\\nBN+{self.activation[:4]}',
                colors['dense'], 0.5, 0.9, size=9
            )
            self._draw_arrow(ax, layer_info[-1][0] + 0.3, layer_info[-1][1],
                           current_x - 0.25, y_center)
            layer_info.append((current_x, y_center))
            current_x += x_spacing
            
            # ========== Output Layer ==========
            self._draw_layer_node(
                ax, current_x, y_center,
                f'Output\\n1 unit\\nSigmoid',
                colors['output'], 0.6, 1.0, size=14
            )
            self._draw_arrow(ax, layer_info[-1][0] + 0.25, layer_info[-1][1],
                           current_x - 0.3, y_center, linewidth=2)
            
            # ========== Title ==========
            title = 'Deep Residual Neural Network Architecture for Skin Lesion Classification'
            ax.text(5, 9.3, title, ha='center', fontsize=14, fontweight='bold')
            
            subtitle = f'Features: {self.input_dim} → Hidden Layers: {self.hidden_units} → Binary Output (BCC/BKL)'
            ax.text(5, 8.8, subtitle, ha='center', fontsize=14, style='italic')
            
            # ========== Architecture Details Box ==========
            details = [
                f'Architecture Components:',
                f'• Residual Blocks: Skip connections for gradient flow',
                f'• Batch Normalization: Training stabilization',
                f'• {self.activation.upper()} Activation: Non-linear transformations',
                f'• Dropout: {self.dropout_rate} (adaptive per layer)',
                f'• Self-Attention: {"Enabled" if self.use_attention else "Disabled"}',
                f'• Regularization: L2 = {self.l2_reg}',
                f'• Loss Function: Focal Loss (α={self.focal_loss_alpha}, γ={self.focal_loss_gamma})',
                f'• Training: Mixup augmentation (α={self.mixup_alpha})'
            ]
            details_text = '\\n'.join(details)
            
            ax.text(0.3, 2.5, details_text, fontsize=14, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=0.5),
                   verticalalignment='top')
            
            # ========== Legend ==========
            legend_elements = [
                mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input'),
                mpatches.Patch(facecolor=colors['dense'], edgecolor='black', label='Dense Layer'),
                mpatches.Patch(facecolor=colors['residual'], edgecolor='black', label='Residual Block'),
            ]
            if self.use_attention:
                legend_elements.append(mpatches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention'))
            legend_elements.append(mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output'))
            
            ax.legend(handles=legend_elements, loc='lower right', fontsize=14, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save figure
            if save_path is None:
                output_dir = Path('output/architecture')
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / 'dnn_architecture_visualization.png'
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"✅ DNN architecture visualization saved to: {save_path}")
            
            plt.close(fig)
            return fig
            
        except Exception as e:
            print(f"❌ Error creating architecture visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _draw_layer_node(self, ax, x, y, text, color, width=0.6, height=0.8, size=11):
        """Draw a layer node as a rounded rectangle"""
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05", 
            facecolor=color, edgecolor='black', linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=size,
               fontweight='bold', color='white')
    
    def _draw_arrow(self, ax, x1, y1, x2, y2, style='solid', linewidth=1.5):
        """Draw an arrow between layers"""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15, 
            linewidth=linewidth, color='black', linestyle=style,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    def _draw_residual_block_compact(self, ax, x, y, units, dropout, prev_info, colors, idx):
        """Draw a compact residual block with skip connection"""
        # Main path (top)
        y_main = y + 0.9
        self._draw_layer_node(
            ax, x, y_main,
            f'{units}',
            colors['residual'], 0.5, 0.6, size=14
        )
        
        # Draw arrows
        prev_x, prev_y = prev_info
        self._draw_arrow(ax, prev_x + 0.3, prev_y, x - 0.25, y_main, 'solid', 1.2)
        
        # Skip connection (curved)
        arrow_skip = FancyArrowPatch(
            (prev_x + 0.3, prev_y), (x + 0.25, y),
            arrowstyle='->', mutation_scale=12,
            linewidth=1.5, color='red', linestyle='dashed',
            connectionstyle="arc3,rad=-.25", alpha=0.6
        )
        ax.add_patch(arrow_skip)
        
        # Add node (merge point)
        self._draw_layer_node(
            ax, x, y,
            '+',
            colors['residual'], 0.35, 0.35, size=14
        )
        self._draw_arrow(ax, x, y_main - 0.3, x, y + 0.18, linewidth=1.2)


# Example usage and testing
if __name__ == '__main__':
    print("TabularDNNClassifier - Modern Deep Learning for Handcrafted Features")
    print("=" * 70)
    
    # Generate synthetic data for testing
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    
    print("\n1. Generating synthetic dataset (511 features, 2000 samples)...")
    X, y = make_classification(
        n_samples=2000,
        n_features=511,
        n_informative=200,
        n_redundant=100,
        n_classes=2,
        weights=[0.84, 0.16],  # Imbalanced like BCC/SK
        random_state=42
    )
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Train model
    print("\n2. Training TabularDNNClassifier...")
    classifier = TabularDNNClassifier(
        hidden_units=[512, 384, 256, 128],
        dropout_rate=0.3,
        learning_rate=1e-3,
        batch_size=64,
        epochs=50,
        patience=15,
        focal_loss_alpha=0.16,
        mixup_alpha=0.2,
        verbose=1,
        random_state=42
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate
    print("\n3. Evaluating on test set...")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    
    print("\n" + "=" * 70)
    print("TabularDNNClassifier test completed successfully!")
