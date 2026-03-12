"""
model.py - Trains LinearRegression and RandomForestRegressor for house price prediction.
"""
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Fit a Linear Regression model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Fitted LinearRegression model.
    """
    logger.info("Training Linear Regression model.")
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("Linear Regression training failed: %s", e)
        raise
    logger.info("Linear Regression training complete.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200) -> RandomForestRegressor:
    """Fit a Random Forest Regressor.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees in the forest.

    Returns:
        Fitted RandomForestRegressor model.
    """
    logger.info("Training Random Forest model. n_estimators=%d", n_estimators)
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion="absolute_error", random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("Random Forest training failed: %s", e)
        raise
    logger.info("Random Forest training complete.")
    return model
