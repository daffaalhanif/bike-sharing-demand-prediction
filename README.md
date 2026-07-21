# 🚲 Bike Sharing Demand Prediction

## 📌 Project Description

This project is an end-to-end machine learning implementation to predict
hourly bike rental demand for the Capital Bikeshare system in Washington
D.C. The model is built to help operators plan bike distribution proactively
and in a data-driven manner, replacing the reactive, intuition-based
approach that is prone to under-supply and over-supply conditions.

## ❗️ Problem Statement

Bike-sharing operators face the challenge of demand uncertainty that
fluctuates dynamically every hour. Without a reliable prediction system,
operators are forced to rely on manual distribution, which is prone to
two main risks:

- **Under-supply** — potential riders are left unserved, increasing churn rate
- **Over-supply** — bikes pile up at certain stations, increasing
  redistribution costs

## 🎯 Objective

Build a regression-based machine learning model capable of accurately
predicting the total hourly bike rental count (`cnt`) based on temporal
and weather features, enabling operators to plan bike distribution
proactively.

**Quantitative targets:**
- RMSE at least 50% better than the Linear Regression baseline
- MAE ≤ 87.63 bikes (50% better than the naive baseline)
- R² above 0.85

## 🔍 Analytical Approach

- **Problem type:** Supervised Learning – Regression
- **Target variable:** `cnt` (total hourly bike rentals)
- **Evaluation metrics:** RMSE is used as the primary metric because it
  penalizes large errors more heavily — fitting for an operational context
  where large-scale prediction errors directly affect bike availability.
  MAE is used as an operational reference since its interpretation in bike
  units is intuitive. R² is used as supplementary information on how well
  the model captures overall data patterns.

## 📊 Dataset

The dataset contains hourly bike rental data from Capital Bikeshare,
Washington D.C., covering the 2011–2012 period, with 12,165 rows and
11 columns.

**Source:** [Capital Bikeshare System Data](http://capitalbikeshare.com/system-data)

## 🤖 Model and Results

Six algorithms were tested and compared — from a Linear Regression
baseline to ensemble models. LightGBM was selected as the final model
after a two-phase hyperparameter tuning process (RandomizedSearch +
BayesSearch).

**How LightGBM works:** The model builds a series of decision trees
incrementally — each new tree focuses on correcting the errors of the
previous ones. LightGBM uses a leaf-wise growth strategy that is more
efficient, allowing it to capture complex non-linear patterns — such as
the bimodal rush-hour pattern — that Linear Regression cannot capture.

| Metric | Target | Result | Status |
|---|---|---|---|
| RMSE | 50% better than LR baseline (164.13) | 64.50 (↓ 61%) | ✅ |
| MAE | ≤ 87.63 bikes | 43.91 (↓ 75% from naive baseline) | ✅ |
| R² | > 0.85 | 0.9144 | ✅ |

## 📝 Conclusion

All three targets were successfully met. The LightGBM model built in this
project predicts hourly bike demand with an average deviation of 44 bikes
from the actual value — a substantial improvement over a no-model approach,
which would miss by an average of 175 bikes per hour. The model is most
reliable under clear-to-light-rain weather conditions (weathersit 1–3)
across all operational time segments.

## 💡 Recommendations

- The model is best used as a planning aid for bike distribution before
  operational hours begin
- Add stock buffer during the morning rush hour (06:00–09:00), as this
  segment has the highest MAE (47.69 bikes)
- The model should be retrained periodically with the latest data as the
  user base grows
- Integrate hourly weather forecasts from an external weather API to
  automate the operational prediction pipeline

## ⚠️ Model Limitations

- The dataset only covers 2011–2012 — the model needs to be retrained
  periodically with more recent data
- Predictions for extreme weather conditions (`weathersit = 4`) are not
  reliable, as the training set contains only 3 rows for this condition
- The model predicts total system-level demand, not per-station demand

## 🛠️ Tech Stack

**Data Manipulation & Analysis:**
- `pandas` — DataFrame manipulation and data aggregation
- `numpy` — array operations and numerical computation

**Visualization:**
- `matplotlib` — canvas layout and chart customization
- `seaborn` — primary statistical visualization

**Machine Learning:**
- `scikit-learn` — preprocessing (StandardScaler, Pipeline),
  cross-validation (TimeSeriesSplit), hyperparameter tuning
  (RandomizedSearchCV), evaluation metrics
- `scikit-optimize` — Bayesian hyperparameter tuning (BayesSearchCV)
- `lightgbm` — final model
- `xgboost` — candidate model for tuning

## 📁 Repository Structure

- `bike_sharing_demand_prediction.ipynb` — Full documentation notebook
- `bike_sharing_demand_lgbm.pkl` — Final model (LightGBM Tuned)
- `data_bike_sharing.csv` — Dataset
- `README.md` — Project introduction
