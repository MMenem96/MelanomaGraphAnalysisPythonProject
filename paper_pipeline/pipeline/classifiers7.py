"""
Multiclass-capable classifier factory for the 7-class arm (Track B).

Two of the nine classifiers in `classifiers.py` are binary-only and failed on
the first 7-class run:

  * **CatBoost** — `eval_metric="F1"` returns one value per class in multiclass
    mode, which CatBoost cannot use for early stopping ("Eval metric should have
    a single value"). The multiclass equivalent is `TotalF1`.
  * **Deep DNN** — `TabularDNNClassifier` hard-codes a single sigmoid output and
    raises "only supports binary classification. Found 7 classes."

Neither is fixed by editing the originals: `classifiers.py` and
`src/tabular_dnn_classifier.py` are the exact code behind the binary paper now
under journal review, and they must keep reproducing its numbers. This module
therefore *wraps* them — every other classifier is delegated untouched.

`DNN7` mirrors the binary net's hyperparameters (same hidden units, dropout,
L2, learning rate, batch size, patience) with a softmax head and class weights,
so the two are comparable by design.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from paper_pipeline.pipeline.classifiers import ACTIVE_CLASSIFIERS, build_classifier

LOG = logging.getLogger("classifiers7")


# Feature values are clipped to this before the float32 cast. RobustScaler can
# emit very large magnitudes on near-constant columns, and casting those to
# float32 overflows to inf, which turns the loss into NaN.
_F32_CLIP = 3.0e38


class _CatBoost7:
    """CatBoost wrapper whose `predict` returns a 1-D label vector.

    In multiclass mode CatBoost returns shape (n, 1); sklearn's metrics expect
    (n,). Left unravelled this silently distorts every per-class metric.
    """

    def __init__(self, model):
        self._model = model

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel().astype(int)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep: bool = True):
        return self._model.get_params(deep)

    def set_params(self, **params):
        self._model.set_params(**params)
        return self


class DNN7:
    """sklearn-compatible softmax MLP, mirroring the binary Deep DNN's settings."""

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_units=(512, 256, 128), dropout_rate: float = 0.2,
                 l2_reg: float = 1e-5, learning_rate: float = 1e-3,
                 batch_size: int = 128, epochs: int = 150, patience: int = 20,
                 validation_split: float = 0.2, random_state: int = 42,
                 verbose: int = 0):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_units = list(hidden_units)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.classes_ = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("input_dim", "n_classes", "hidden_units", "dropout_rate", "l2_reg",
                 "learning_rate", "batch_size", "epochs", "patience",
                 "validation_split", "random_state", "verbose")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        tf.random.set_seed(self.random_state)
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.hidden_units:
            x = layers.Dense(units, activation="relu",
                             kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf

        X = np.clip(np.asarray(X, dtype=np.float64), -_F32_CLIP, _F32_CLIP).astype(np.float32)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.model_ = self._build()

        # Class weights so rare classes are not drowned out even when the
        # training frame is not perfectly balanced.
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = len(y) / (self.n_classes * counts)
        class_weight = {i: float(w) for i, w in enumerate(weights)}

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)]
        self.model_.fit(
            X, y,
            batch_size=self.batch_size, epochs=self.epochs,
            validation_split=self.validation_split,
            class_weight=class_weight, callbacks=callbacks, verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("DNN7 is not fitted")
        Xc = np.clip(np.asarray(X, dtype=np.float64), -_F32_CLIP, _F32_CLIP).astype(np.float32)
        return self.model_.predict(Xc, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build_classifier7(name: str, input_dim: int | None = None, n_classes: int = 7):
    """Multiclass-safe classifier factory. Non-patched names delegate unchanged."""
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        params = dict(ACTIVE_CLASSIFIERS["CatBoost"]["params"])
        # 'F1' yields one value per class in multiclass mode and cannot drive
        # early stopping; 'TotalF1' is the aggregate equivalent.
        params["eval_metric"] = "TotalF1"
        params["loss_function"] = "MultiClass"
        return _CatBoost7(CatBoostClassifier(**params))

    if name == "Deep DNN":
        if input_dim is None:
            raise ValueError("Deep DNN needs input_dim")
        p = ACTIVE_CLASSIFIERS["Deep DNN"]["params"]
        return DNN7(
            input_dim=input_dim, n_classes=n_classes,
            hidden_units=p["hidden_units"], dropout_rate=p["dropout_rate"],
            l2_reg=p["l2_reg"], learning_rate=p["learning_rate"],
            batch_size=p["batch_size"], epochs=p["epochs"], patience=p["patience"],
            validation_split=p["validation_split"], random_state=p["random_state"],
            verbose=p["verbose"],
        )

    return build_classifier(name, input_dim=input_dim)
