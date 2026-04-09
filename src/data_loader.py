# data_loader.py - this file loads the real estate dataset from a csv and checks that it's valid
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# these are all the columns our dataset needs to have for the model to work
REQUIRED_COLUMNS = [
    "price", "year_sold", "property_tax", "insurance", "beds", "baths",
    "sqft", "year_built", "lot_size", "basement", "popular", "recession",
    "property_age", "property_type_Condo",
]


def load_data(filepath: str) -> pd.DataFrame:
    # load the csv file and make sure it has all the columns we need
    logger.info("Loading data from: %s", filepath)
    try:
        df = pd.read_csv(filepath)  # read the csv into a pandas dataframe
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise
    # make sure the dataframe isn't empty
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    # check if any required columns are missing from the dataset
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info("Data loaded successfully. Shape: %s", df.shape)
    return df
