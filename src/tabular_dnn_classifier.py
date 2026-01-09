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
