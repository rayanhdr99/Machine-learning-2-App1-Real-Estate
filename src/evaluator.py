"""
evaluator.py - Evaluates regression models using Mean Absolute Error.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def evaluate_model(model, X, y, dataset_name: str = "Test") -> float:
    """Compute and log MAE for a given dataset.

    Args:
        model: Trained scikit-learn regressor.
        X: Feature matrix.
        y: True target values.
        dataset_name: Label used in log messages.

    Returns:
        Mean Absolute Error as a float.
    """
    try:
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", dataset_name, e)
        raise
    logger.info("%s MAE: %.2f", dataset_name, mae)
    return mae


def feature_importance(model, feature_columns: list) -> pd.DataFrame:
    """Return feature importances for a RandomForestRegressor.

    Args:
        model: Fitted RandomForestRegressor.
        feature_columns: List of feature names.

    Returns:
        DataFrame sorted by importance descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not support feature_importances_.")
    df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
