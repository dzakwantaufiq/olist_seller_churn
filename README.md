# 🛒 Olist E-Commerce Seller Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-Spatial-yellow.svg)](https://duckdb.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Olist%20Brazilian%20Ecommerce-blue)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Geospatial Analytics pipeline designed to predict seller churn on **Olist**, Brazil's largest marketplace. By combining SQL-based feature store generation, geospatial boundary filtering, and machine learning, this project identifies high-risk sellers to enable proactive marketplace retention strategies.

---

## 📌 Business Overview & Problem Statement

For two-sided e-commerce platforms like Olist, merchant churn directly impacts product variety, customer fulfillment, and marketplace revenue. 

**Objectives:**
1. **Define Mathematical Churn:** Establish a dynamic observation window based on historical carrier dispatch dates.
2. **Geospatial Data Sanitization:** Clean invalid geographic coordinates using Brazilian boundary shapefiles via DuckDB Spatial.
3. **Feature Engineering:** Aggregate RFM (Recency, Frequency, Monetary), logistical distance, revenue momentum, and consistency metrics.
4. **Deployable Machine Learning:** Train and export a robust, production-ready Scikit-Learn `Pipeline` for real-time risk scoring.

---

## 🏗️ System Architecture & Workflow

```text
┌─────────────────────────┐
│ Kaggle Olist Dataset    │ (9 Relational CSVs)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ DuckDB Spatial Processing│ ──► Filters lat/lng via geoBoundaries BRA ADM0
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ SQL Feature Store       │ ──► Derives RFM, Distance, Momentum & Consistency
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Preprocessing Pipeline  │ ──► SimpleImputer(median) + RobustScaler
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Model Training & Eval   │ ──► Logistic Regression / Ensembles
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Serialized Pipeline     │ ──► Exported as model_seller_churn.sav
└─────────────────────────┘
```

---

## 🎯 Churn Definition & Data Filtering

Churn is defined dynamically based on carrier dispatch dates:
- **Cutoff Date ($T_{\\text{cutoff}}$):** 30 days prior to the latest recorded order carrier dispatch date ($T_{\\text{max}} - 30\\text{ days}$).
- **Target Label Mapping:**
  - **Active (`0`):** Seller dispatched at least one order on or after $T_{\\text{cutoff}}$.
  - **Churn (`1`):** Seller dispatched orders prior to $T_{\\text{cutoff}}$ but remained inactive in the subsequent 30-day window.
  - **Not Eligible (Excluded):** Sellers with zero total orders or whose first order occurred within 30 days of $T_{\\text{cutoff}}$.

---

## 📐 Feature Dictionary (Model Schema)

The exported pipeline (`model_seller_churn.sav`) operates on the following 9 numerical features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `total_orders` | Integer | Total lifetime orders fulfilled by the seller. |
| `days_since_last_order` | Integer | Recency count of days between last order and cutoff date. |
| `active_months` | Integer | Count of distinct calendar months with active orders. |
| `orders_last_90d` | Integer | Order volume fulfilled in the 90 days prior to cutoff. |
| `gmv_prev_30d` | Float | Gross Merchandise Value ($) generated in the previous 30-day window. |
| `n_categories` | Integer | Diversity count of distinct product categories sold. |
| `max_distance_km` | Float | Maximum geographic fulfillment distance between seller and buyer. |
| `momentum_orders` | Float | Order growth trend ($\\text{Orders}_{\\text{last 30d}} / \\text{Orders}_{\\text{prev 30d}}$). |
| `consistency_ratio` | Float | Ratio of active months relative to overall seller tenure. |

---

## 🔬 Machine Learning Pipeline

The serialized pipeline (`model_seller_churn.sav`) packages preprocessing and model evaluation into a single atomic artifact:

1. **Preprocessing (`ColumnTransformer`):**
   - **`SimpleImputer`:** Imputes missing values using the `median` strategy.
   - **`RobustScaler`:** Scales features using quantiles to remain resilient against extreme revenue or order outliers.
2. **Classifier:**
   - **`LogisticRegression`:** L-BFGS solver calibrated for probability outputs.
   - Evaluated alongside Decision Trees, Random Forests, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost using Recall and $F_\\beta$-Score metrics to minimize undetected churn.

---

## 📂 Repository Structure

```text
.
├── olist_seller_churn.ipynb  # Complete notebook (EDA, DuckDB Spatial, Modeling)
├── model_seller_churn.sav    # Serialized scikit-learn Pipeline
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/olist-seller-churn.git
cd olist-seller-churn

pip install pandas numpy duckdb geopandas scikit-learn xgboost lightgbm catboost python-dotenv lonboard
```

### 2. DuckDB Spatial Setup

Ensure DuckDB Spatial extension is loaded in Python when running custom queries:

```python
import duckdb
duckdb.sql("INSTALL spatial; LOAD spatial;")
```

---

## 💻 Running Inference

You can load `model_seller_churn.sav` directly to score new seller data:

```python
import pickle
import pandas as pd

# Load the saved Scikit-Learn Pipeline
with open('model_seller_churn.sav', 'rb') as f:
    model_pipeline = pickle.load(f)

# Input dataframe using the exact 9 training features
seller_data = pd.DataFrame([{
    'total_orders': 24,
    'days_since_last_order': 45,
    'active_months': 5,
    'orders_last_90d': 2,
    'gmv_prev_30d': 114.17,
    'n_categories': 2,
    'max_distance_km': 1493.66,
    'momentum_orders': 0.5,
    'consistency_ratio': 0.45
}])

# Predict churn class and risk probability
churn_pred = model_pipeline.predict(seller_data)[0]
churn_proba = model_pipeline.predict_proba(seller_data)[0][1]

print(f"Status: {'CHURN RISK' if churn_pred == 1 else 'ACTIVE'}")
print(f"Risk Probability: {churn_proba:.2%}")
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
