"""Model Registry for CML Elimination Prediction.

This module provides a centralized registry of ML models with their
hyperparameter grids for model comparison and selection.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Any, Type
import logging

logger = logging.getLogger(__name__)


# Model definitions with hyperparameter grids for tuning
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "class": RandomForestClassifier,
        "default_params": {"random_state": 42},
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        },
        "description": "Random Forest ensemble classifier",
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "default_params": {"random_state": 42},
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__min_samples_split": [2, 5],
        },
        "description": "Gradient Boosting classifier",
    },
}

# XGBoost is optional - add if available
try:
    from xgboost import XGBClassifier

    MODEL_REGISTRY["xgboost"] = {
        "class": XGBClassifier,
        "default_params": {
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        },
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
        },
        "description": "XGBoost gradient boosting classifier",
    }
    logger.info("XGBoost available and registered")
except ImportError:
    logger.warning("XGBoost not available - install with: pip install xgboost")


def get_model_class(model_name: str) -> Type:
    """Get model class by name.

    Args:
        model_name: Name of the model (random_forest, gradient_boosting, xgboost)

    Returns:
        Model class

    Raises:
        ValueError: If model name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_REGISTRY[model_name]["class"]


def get_param_grid(model_name: str) -> Dict[str, Any]:
    """Get hyperparameter grid for model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of hyperparameter options for grid search
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["param_grid"]


def get_default_params(model_name: str) -> Dict[str, Any]:
    """Get default parameters for model instantiation.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of default parameters
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["default_params"]


def list_available_models() -> Dict[str, str]:
    """List all available models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {name: info["description"] for name, info in MODEL_REGISTRY.items()}
