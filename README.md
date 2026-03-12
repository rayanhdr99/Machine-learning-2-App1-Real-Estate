# Real Estate Price Predictor

A production-quality machine learning web application that predicts residential
property prices using **Linear Regression** and **Random Forest Regressor**,
served through an interactive **Streamlit** dashboard.

---

## Description

This project trains two regression models on a labelled real estate dataset and
exposes them via a multi-page Streamlit UI. Users can:

- Browse a **Dataset Overview** with summary statistics, a price distribution
  histogram, and a full correlation heatmap.
- Compare both models on a dedicated **Model Performance** page showing Train
  and Test MAE metrics, a grouped bar chart, and a Random Forest feature
  importance chart.
- Obtain an instant price estimate on the **Predict Price** page by filling in
  property details and choosing a model.

All data loading, preprocessing, modelling, and evaluation logic is cleanly
separated into a `src/` package with structured logging throughout.

---

## Project Structure

```
real_estate_price_predictor/
├── data/
│   └── final.csv                  # Raw dataset (1860 records, 14 features + target)
├── src/
│   ├── __init__.py                # Package marker
│   ├── data_loader.py             # load_data() - reads and validates the CSV
│   ├── preprocessor.py            # split_data() - stratified train/test split
│   ├── model.py                   # train_linear_regression(), train_random_forest()
│   └── evaluator.py               # evaluate_model(), feature_importance()
├── app.py                         # Streamlit entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Dataset

**File:** `data/final.csv`
**Records:** 1860
**Format:** CSV, read with `pandas.read_csv()`

| Column | Type | Description |
|---|---|---|
| `price` | float | Sale price in USD (target variable) |
| `year_sold` | int | Year the property was sold |
| `property_tax` | float | Annual property tax in USD |
| `insurance` | float | Annual homeowner insurance cost in USD |
| `beds` | int | Number of bedrooms |
| `baths` | int | Number of bathrooms |
| `sqft` | float | Living area in square feet |
| `year_built` | int | Year the property was constructed |
| `lot_size` | float | Total lot size in square feet |
| `basement` | int | Has basement: 1 = Yes, 0 = No |
| `popular` | int | Popular neighbourhood: 1 = Yes, 0 = No |
| `recession` | int | Sold during recession: 1 = Yes, 0 = No |
| `property_age` | float | Age of property at time of sale |
| `property_type_Condo` | int | Property type: 1 = Condo, 0 = House |

---

## Models Used

### 1. Linear Regression
- `sklearn.linear_model.LinearRegression`
- Baseline model; fast to train and interpretable.

### 2. Random Forest Regressor
- `sklearn.ensemble.RandomForestRegressor`
- Configuration: `n_estimators=200`, `criterion='absolute_error'`, `random_state=42`
- Primary prediction model; robust to outliers and provides feature importances.

**Evaluation metric:** Mean Absolute Error (MAE) on the held-out test set.

**Train/test split:** 80/20, stratified on `property_type_Condo` to preserve
class balance in both sets.

---

## How to Run Locally

### Prerequisites

- Python 3.9 or newer
- pip

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd real_estate_price_predictor
```

Or navigate to the folder if you received it as a ZIP.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify the Dataset

Ensure the dataset is in place:

```
real_estate_price_predictor/
    data/
        final.csv    <-- must exist
```

### 4. Launch the App

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) - open it
in your browser.

---

## Dependencies

| Package | Min Version | Purpose |
|---|---|---|
| `streamlit` | 1.32.0 | Web application framework |
| `pandas` | 2.0.0 | Data loading and manipulation |
| `numpy` | 1.24.0 | Numerical operations |
| `scikit-learn` | 1.3.0 | ML models and evaluation metrics |
| `matplotlib` | 3.7.0 | Charts and visualisations |
| `seaborn` | 0.12.0 | Correlation heatmap styling |

---

## Deployment

See Streamlit Cloud deployment link
