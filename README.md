# IDX-Exchange-Data-Science

Predicting California single-family home **ClosePrice** from CRMLS sold listings (IDX Exchange internship, team ds76).

| Week | File | What |
|---|---|---|
| 1 | `column_notes.md`, `dataset_columns.docx` | Dataset access and what each key column means |
| 2 | `01_exploration.ipynb`, `01_exploration_full.ipynb` | EDA: distributions, missing values, geography, trends, correlations |
| 3 | `02_preprocessing.ipynb` | Cleaning, missing values, encoding, scaling, time-based train/test split with a tuned training window |
| 3 | `02_preprocessing_simple.ipynb` | Simpler week 3 version: median fill + missing flags, one-hot county, smoothed ZIP price, log + standard scaling, time split with X = 12 months. Writes the `cleaned_sfr.csv` that week 4 uses |
| 4 | `03_baseline_model.ipynb` | Linear Regression baseline: **test R² = 0.779** on May 2026, median error 12.1%. Run `02_preprocessing_simple.ipynb` first |
| – | `preprocessing.py` | The same week 3 steps as reusable functions, for later notebooks |

## Run it
1. Download the `CRMLSSold*.csv` files from the IDX FTP (`raw/California`) into this folder. They are git-ignored and not committed.
2. `pip install -r requirements.txt`
3. Open the notebooks in order. `02_preprocessing.ipynb` writes `cleaned_sfr.csv`, which later weeks load.
