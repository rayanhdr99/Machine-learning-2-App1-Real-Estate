"""
app.py - Streamlit application for Real Estate Price Prediction.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.preprocessor import split_data
from src.model import train_linear_regression, train_random_forest
from src.evaluator import evaluate_model, feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final.csv")

st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠", layout="wide")


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_resource
def get_models():
    df = get_data()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    lr = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    return lr, rf, X_train, X_test, y_train, y_test, feature_cols


def main():
    st.title("🏠 Real Estate Price Predictor")
    st.markdown("Predict property prices using **Linear Regression** and **Random Forest** models.")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Performance", "Predict Price"])

    try:
        df = get_data()
        lr, rf, X_train, X_test, y_train, y_test, feature_cols = get_models()
    except Exception as e:
        st.error(f"Failed to load data or train models: {e}")
        logger.error("App startup error: %s", e)
        return

    if page == "Dataset Overview":
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Features", df.shape[1] - 1)
        col3.metric("Avg Price", f"${df['price'].mean():,.0f}")

        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["price"], bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Property Prices")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    elif page == "Model Performance":
        st.header("Model Performance Comparison")

        lr_train_mae = evaluate_model(lr, X_train, y_train, "LR Train")
        lr_test_mae = evaluate_model(lr, X_test, y_test, "LR Test")
        rf_train_mae = evaluate_model(rf, X_train, y_train, "RF Train")
        rf_test_mae = evaluate_model(rf, X_test, y_test, "RF Test")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Linear Regression")
            st.metric("Train MAE", f"${lr_train_mae:,.0f}")
            st.metric("Test MAE", f"${lr_test_mae:,.0f}")
        with col2:
            st.subheader("Random Forest")
            st.metric("Train MAE", f"${rf_train_mae:,.0f}")
            st.metric("Test MAE", f"${rf_test_mae:,.0f}")

        # Bar chart comparison
        st.subheader("MAE Comparison")
        results = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Train MAE": [lr_train_mae, rf_train_mae],
            "Test MAE": [lr_test_mae, rf_test_mae],
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2)
        bars1 = ax.bar(x - 0.2, results["Train MAE"], 0.4, label="Train MAE", color="steelblue")
        bars2 = ax.bar(x + 0.2, results["Test MAE"], 0.4, label="Test MAE", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(results["Model"])
        ax.set_ylabel("MAE ($)")
        ax.legend()
        ax.set_title("Train vs Test MAE by Model")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Random Forest Feature Importances")
        fi = feature_importance(rf, feature_cols)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi["Feature"][::-1], fi["Importance"][::-1], color="teal")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances (Random Forest)")
        st.pyplot(fig)
        plt.close(fig)

    elif page == "Predict Price":
        st.header("Predict Property Price")
        st.markdown("Enter the property details below to get a price prediction.")

        col1, col2, col3 = st.columns(3)
        with col1:
            year_sold = st.number_input("Year Sold", min_value=2000, max_value=2030, value=2012)
            property_tax = st.number_input("Property Tax ($/yr)", min_value=0, max_value=10000, value=216)
            insurance = st.number_input("Insurance ($/yr)", min_value=0, max_value=5000, value=74)
            beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        with col2:
            sqft = st.number_input("Square Footage", min_value=100, max_value=20000, value=1500)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1990)
            lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=100000, value=5000)
            basement = st.selectbox("Basement", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col3:
            popular = st.selectbox("Popular Neighborhood", [0, 1], format_func=lambda x: "Yes" if x else "No")
            recession = st.selectbox("Recession Year", [0, 1], format_func=lambda x: "Yes" if x else "No")
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=200, value=20)
            property_type_condo = st.selectbox("Property Type", [0, 1], format_func=lambda x: "Condo" if x else "House")

        model_choice = st.radio("Select Model", ["Linear Regression", "Random Forest"])

        if st.button("Predict Price", type="primary"):
            input_data = pd.DataFrame([{
                "year_sold": year_sold, "property_tax": property_tax, "insurance": insurance,
                "beds": beds, "baths": baths, "sqft": sqft, "year_built": year_built,
                "lot_size": lot_size, "basement": basement, "popular": popular,
                "recession": recession, "property_age": property_age,
                "property_type_Condo": property_type_condo,
            }])
            try:
                model = lr if model_choice == "Linear Regression" else rf
                prediction = model.predict(input_data)[0]
                st.success(f"### Predicted Price: **${prediction:,.2f}**")
                logger.info("Prediction made: $%.2f using %s", prediction, model_choice)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error("Prediction error: %s", e)


if __name__ == "__main__":
    main()
