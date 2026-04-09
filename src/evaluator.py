# evaluator.py - evaluates the models using mean absolute error and shows feature importances
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def evaluate_model(model, X, y, dataset_name: str = "Test") -> float:
    # calculate mean absolute error to see how off our predictions are
    try:
        y_pred = model.predict(X)  # get the model's predictions
        mae = mean_absolute_error(y, y_pred)  # compare predictions to actual values
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", dataset_name, e)
        raise
    logger.info("%s MAE: %.2f", dataset_name, mae)
    return mae


def feature_importance(model, feature_columns: list) -> pd.DataFrame:
    # get feature importances from the random forest and return them sorted
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not support feature_importances_.")
    # build a dataframe with feature names and their importance scores, sorted highest first
    df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
