"""
The 9 classifiers reported in the paper abstract.

Hyperparameters are copied verbatim from manual_train_features.py:59-272
so the new pipeline reproduces the old behaviour for these specific models.
The 6 classifiers excluded from the abstract (XGBoost, RF, SVM-Linear,
SVM-Sigmoid, SVM-Poly, Calibrated SVM) are intentionally not built here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.tabular_dnn_classifier import TabularDNNClassifier


# The 9 active classifiers, in the same order the paper abstract lists them.
ACTIVE_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "SVM (RBF)": {
        "class": SVC,
        "params": {
            "C": 100,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "cache_size": 1000,
            "probability": True,
            "random_state": 42,
        },
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "force_col_wise": True,
            "feature_pre_filter": False,
        },
    },
    "CatBoost": {
        "class": CatBoostClassifier,
        "params": {
            "iterations": 1000,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "random_strength": 1.5,
            "bagging_temperature": 1.0,
            "border_count": 128,
            "random_state": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
            "task_type": "CPU",
            "eval_metric": "F1",
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
        },
    },
    "Extra Trees": {
        "class": ExtraTreesClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 25,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "euclidean",
            "algorithm": "auto",
            "n_jobs": -1,
        },
    },
    "Logistic Regression": {
        "class": CalibratedClassifierCV,
        "params": {
            "estimator": LogisticRegression(
                C=10.0,
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            ),
            "method": "sigmoid",
            "cv": 5,
        },
    },
    "MLP": {
        "class": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (100, 50),
            "alpha": 0.001,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        },
    },
    "Deep DNN": {
        "class": TabularDNNClassifier,
        "params": {
            "input_dim": 511,  # rewritten dynamically before fit
            "hidden_units": [512, 256, 128],
            "dropout_rate": 0.2,
            "l2_reg": 1e-5,
            "activation": "relu",
            "use_attention": False,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "epochs": 150,
            "patience": 20,
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "mixup_alpha": 0.0,
            "validation_split": 0.2,
            "verbose": 0,
            "random_state": 42,
        },
    },
}


def build_classifier(name: str, input_dim: int | None = None):
    """Instantiate a fresh classifier by name. `input_dim` only matters for Deep DNN."""
    if name not in ACTIVE_CLASSIFIERS:
        raise KeyError(
            f"Unknown classifier {name!r}. Active classifiers: {list(ACTIVE_CLASSIFIERS)}"
        )
    spec = ACTIVE_CLASSIFIERS[name]
    cls = spec["class"]
    params = dict(spec["params"])
    if name == "Deep DNN" and input_dim is not None:
        params["input_dim"] = input_dim
    return cls(**params)
