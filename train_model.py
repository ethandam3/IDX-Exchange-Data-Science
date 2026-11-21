"""Week 9: train the model behind app.py and save it to model.joblib.

Run once (needs cleaned_sfr.csv from 02_preprocessing_simple.ipynb):
    python train_model.py
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

TARGET = "ClosePrice"
INPUTS = ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet"]
FEATURES = INPUTS + ["ZipLat", "ZipLon", "ZipPrice", "County"]
ZIP_SMOOTHING = 20      # same smoothing as Week 3
MIN_ZIP_SALES = 5       # ZIPs offered in the app's dropdown
COVERAGE = 0.90
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=134, min_child_samples=25, subsample=0.784,
                  subsample_freq=1, colsample_bytree=0.313, reg_lambda=0.295, n_estimators=1054,
                  verbose=-1, random_state=0)   # Week 7 Optuna settings

df = pd.read_csv("cleaned_sfr.csv", parse_dates=["CloseDate"], dtype={"PostalCode": str})
df["Month"] = df["CloseDate"].dt.to_period("M")

# ZIP centre and county use location only (no prices), so they can come from every row
zip_info = df.groupby("PostalCode").agg(ZipLat=("Latitude", "median"), ZipLon=("Longitude", "median"),
                                        County=("CountyOrParish", lambda s: s.mode().iloc[0]))
COUNTIES = sorted(zip_info["County"].unique())


def window(end_month, months=12):
    """The `months` months before `end_month` (exclusive)."""
    end = pd.Period(end_month, "M")
    return df[(df["Month"] >= end - months) & (df["Month"] < end)]


def zip_prices(train):
    """Smoothed average log price per ZIP, learned from training rows only."""
    log_price = np.log(train[TARGET])
    overall = log_price.mean()
    stats = log_price.groupby(train["PostalCode"]).agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + ZIP_SMOOTHING * overall) / (stats["count"] + ZIP_SMOOTHING)
    return smoothed, overall, stats["count"]


def features(rows, prices, overall, medians):
    X = rows[INPUTS].fillna(medians).copy()
    for col in ["LivingArea", "LotSizeSquareFeet"]:
        X[col] = np.log1p(X[col])
    info = zip_info.reindex(rows["PostalCode"].values)
    X["ZipLat"], X["ZipLon"] = info["ZipLat"].values, info["ZipLon"].values
    X["ZipPrice"] = rows["PostalCode"].map(prices).fillna(overall).values
    X["County"] = pd.Categorical(info["County"].values, categories=COUNTIES)
    return X[FEATURES]


def fit(train):
    prices, overall, counts = zip_prices(train)
    medians = train[INPUTS].median()
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(features(train, prices, overall, medians), np.log(train[TARGET]))
    return dict(model=model, prices=prices, overall=overall, medians=medians, counts=counts)


def predict(m, rows):
    return np.exp(m["model"].predict(features(rows, m["prices"], m["overall"], m["medians"])))


def range_widths(m, holdout):
    """Conformal-style widths: how far off (in log price) the model was on a month it never saw."""
    log_err = np.log(holdout[TARGET].values) - np.log(predict(m, holdout))
    tail = (1 - COVERAGE) / 2
    return np.quantile(log_err, tail), np.quantile(log_err, 1 - tail)


# 1) widths from April 2026, using a model trained on the 12 months before it
april = df[df["Month"] == pd.Period("2026-04", "M")]
lo, hi = range_widths(fit(window("2026-04")), april)

# 2) honest test on May 2026 (same split as every other week)
may = df[df["Month"] == pd.Period("2026-05", "M")]
m = fit(window("2026-05"))
pred = predict(m, may)
inside = (may[TARGET].values >= pred * np.exp(lo)) & (may[TARGET].values <= pred * np.exp(hi))
metrics = {"test_R2": r2_score(may[TARGET], pred),
           "median_pct_error": float(np.median(np.abs(pred - may[TARGET].values) / may[TARGET].values)),
           "range_coverage": float(inside.mean())}
print(f"May 2026 test: R² {metrics['test_R2']:.3f}, typical error {metrics['median_pct_error']:.1%}, "
      f"90% range coverage {metrics['range_coverage']:.1%}")
assert 0.85 <= metrics["range_coverage"] <= 0.95, "price ranges are badly calibrated"

# 3) final model on the newest 12 months; its range widths come from May
lo, hi = range_widths(m, may)
final = fit(window("2026-06"))
zips = final["counts"][final["counts"] >= MIN_ZIP_SALES].index
final.update(zip_info=zip_info.loc[zips], counties=COUNTIES, inputs=INPUTS, range=(lo, hi),
             metrics=metrics, trained_on="Jun 2025 – May 2026")
joblib.dump(final, "model.joblib")
print(f"Saved model.joblib: {len(zips):,} ZIPs, 90% range = {np.exp(lo) - 1:+.0%} / {np.exp(hi) - 1:+.0%}")
