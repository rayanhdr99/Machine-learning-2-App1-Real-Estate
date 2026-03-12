"""
preprocessor.py - Splits data into train/test sets for the Real Estate model.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Separate features/target and split into train/test sets.

    Args:
        df: Loaded real estate DataFrame.
        test_size: Fraction for the test set.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_columns).
    """
    logger.info("Splitting data. Test size: %.0f%%", test_size * 100)
    X = df.drop("price", axis=1)
    y = df["price"]
    feature_columns = list(X.columns)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=X["property_type_Condo"]
        )
    except Exception as e:
        logger.error("Train/test split failed: %s", e)
        raise
    logger.info("Train size: %d  Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, feature_columns
