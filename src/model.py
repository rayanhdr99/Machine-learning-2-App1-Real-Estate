# model.py - trains the linear regression and random forest models for price prediction
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_linear_regression(X_train, y_train) -> LinearRegression:
    # train a linear regression model on the training data
    logger.info("Training Linear Regression model.")
    try:
        model = LinearRegression()  # create a basic linear regression model
        model.fit(X_train, y_train)  # fit the model to the training data
    except Exception as e:
        logger.error("Linear Regression training failed: %s", e)
        raise
    logger.info("Linear Regression training complete.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 200) -> RandomForestRegressor:
    # train a random forest model with 200 trees using MAE as the criterion
    logger.info("Training Random Forest model. n_estimators=%d", n_estimators)
    try:
        # create the random forest with absolute error criterion and a fixed random state for reproducibility
        model = RandomForestRegressor(n_estimators=n_estimators, criterion="absolute_error", random_state=42)
        model.fit(X_train, y_train)  # fit the model to the training data
    except Exception as e:
        logger.error("Random Forest training failed: %s", e)
        raise
    logger.info("Random Forest training complete.")
    return model
