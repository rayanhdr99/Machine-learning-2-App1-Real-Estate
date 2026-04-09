# preprocessor.py - splits the dataset into training and testing sets
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    # separate features and target, then split into 80% training and 20% testing
    logger.info("Splitting data. Test size: %.0f%%", test_size * 100)
    X = df.drop("price", axis=1)  # drop the price column to get features only
    y = df["price"]  # separate the price column as the target variable
    feature_columns = list(X.columns)  # save feature names for later use
    try:
        # use stratified split on property type so both sets have similar distributions
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=X["property_type_Condo"]
        )
    except Exception as e:
        logger.error("Train/test split failed: %s", e)
        raise
    logger.info("Train size: %d  Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, feature_columns
