# IDX-Exchange-Data-Science

Predicting California single-family home **ClosePrice** from CRMLS sold listings (IDX Exchange internship, team ds76).

| Week | File | What |
|---|---|---|
| 1 | `column_notes.md`, `dataset_columns.docx` | Dataset access and what each key column means |
| 2 | `01_exploration.ipynb`, `01_exploration_full.ipynb` | EDA: distributions, missing values, geography, trends, correlations |
| 3 | `02_preprocessing.ipynb` | Cleaning, missing values, encoding, scaling, time-based train/test split with a tuned training window |
| 3 | `02_preprocessing_simple.ipynb` | Simpler week 3 version: median fill + missing flags, one-hot county, smoothed ZIP price, log + standard scaling, time split with X = 12 months. Writes the `cleaned_sfr.csv` that week 4 uses |
| 4 | `03_baseline_model.ipynb` | Linear Regression baseline: **test R² = 0.779** on May 2026, median error 12.1%. Run `02_preprocessing_simple.ipynb` first |
| 5 | `04_model_comparison.ipynb` | Decision Tree (depth tuned on April 2026) and Random Forest vs the baseline. Random Forest: **test R² = 0.854**, median error 8.2% |
| 6 | `05_feature_engineering.ipynb` | Age and ratio features, school district (GeoPandas spatial join), nearby comps (BallTree). Random Forest: **test R² = 0.883**. Writes `features_sfr.csv` |
| 7 | `06_advanced_models.ipynb` | XGBoost (grid search) and LightGBM (Optuna). Best: Optuna-tuned LightGBM, **test R² = 0.885**, median error 7.8% |
| 8 | `07_evaluation.ipynb` | MAE, RMSE, MAPE, errors by price band and county, 90% conformal price ranges, 12-month walk-forward backtest. Writes `metrics_summary.csv` |
| 9 | `train_model.py`, `app.py` | Streamlit price app: living area, beds, baths, lot size and ZIP → predicted price with a 90% range. LightGBM on 5 inputs: **test R² = 0.859** on May 2026, median error 8.9%, range coverage 90.0% |
| – | `preprocessing.py` | Functions from the earlier, more advanced week 3 version |
| – | `DistrictAreas2425.zip` | California School District Areas 2024-25 boundaries, used by week 6 |

## Run it
1. Download the `CRMLSSold*.csv` files from the IDX FTP (`raw/California`) into this folder. They are git-ignored and not committed.
2. `pip install -r requirements.txt`
3. Run the notebooks in order, `02_preprocessing_simple.ipynb` through `07_evaluation.ipynb`. `02_preprocessing_simple.ipynb` writes `cleaned_sfr.csv` (used by weeks 4–6), and `05_feature_engineering.ipynb` writes `features_sfr.csv` (used by weeks 7–8).
4. Week 9 app: `python train_model.py` (writes `model.joblib`, about 30 s), then `streamlit run app.py`.
