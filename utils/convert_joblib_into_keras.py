import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the scikit-learn model and scaler
MODEL_PATH = "melanoma_classifier.joblib"
SCALER_PATH = "scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define a simple TensorFlow model (equivalent to Scikit-Learn SVM/RF)
tf_model = keras.Sequential([
    keras.layers.Input(shape=(model.n_features_in_,)),  # Input shape matches sklearn model
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification (Melanoma vs Benign)
])

# Compile the TensorFlow model
tf_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Convert sklearn model predictions into TensorFlow format
X_train = np.random.rand(100, model.n_features_in_)  # Simulated training data
y_train = model.predict(X_train)  # Use the original scikit-learn model to label data

# Train the TensorFlow model on generated data
tf_model.fit(X_train, y_train, epochs=10, batch_size=8)

# Save TensorFlow model
tf_model.save("melanoma_tf_model")
