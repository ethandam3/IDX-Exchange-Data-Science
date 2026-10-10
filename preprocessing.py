"""Week 3 preprocessing for the CRMLS single-family ClosePrice model (team ds76).

Two kinds of steps:
  * clean()              - fixed rules (filters, dedupe, bad values -> NaN). Safe to run on all months.
  * build_preprocessor() - learned steps (imputation, encoding, scaling). Fit on the TRAINING window only,
                           so nothing from the test month leaks into training.
Later notebooks (week 4+) can just `from preprocessing import *`.
"""
import glob
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

TARGET = "ClosePrice"
SKEWED = ["LivingArea", "LotSizeSquareFeet", "AssociationFeeMonthly"]          # log1p before scaling
NUMERIC = ["BedroomsTotal", "BathroomsTotalInteger", "YearBuilt", "GarageSpaces", "ParkingTotal",
           "Stories", "MainLevelBedrooms", "Latitude", "Longitude"]
BINARY = ["ViewYN", "PoolPrivateYN", "FireplaceYN", "AttachedGarageYN", "NewConstructionYN"]
LOW_CARD = ["CountyOrParish", "Levels"]                                          # one-hot
HIGH_CARD = ["City", "PostalCode", "MLSAreaMajor", "HighSchoolDistrict"]         # target-encoded
FLOORING = ["Tile", "Carpet", "Wood", "Laminate", "Vinyl", "Stone"]              # tokens in >=2% of listings
FLOORING_COLS = [f"Floor_{t}" for t in FLOORING]
FEATURES = SKEWED + NUMERIC + BINARY + LOW_CARD + HIGH_CARD + FLOORING_COLS

CA_LAT, CA_LON = (32.4, 42.1), (-124.5, -114.1)
HOA_PER_MONTH = {"Monthly": 1, "Quarterly": 1 / 3, "SemiAnnually": 1 / 6, "Annually": 1 / 12}


def load_raw(folder="."):
    """Concatenate every CRMLSSold*.csv (incl. the *_filled.csv versions) in `folder`."""
    frames = []
    for f in sorted(glob.glob(f"{folder}/CRMLSSold*.csv")):
        d = pd.read_csv(f, low_memory=False)
        d["SourceMonth"] = re.search(r"CRMLSSold(\d{6})", f).group(1)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def clean(raw, verbose=True):
    """Rule-based cleaning. Returns one row per sale with ListingKey, CloseDate, TARGET and FEATURES."""
    log = []
    df = raw[(raw["PropertyType"] == "Residential") &
             (raw["PropertySubType"] == "SingleFamilyResidence")].copy()
    log.append(("SFR filter", len(df)))

    # the same listing can show up in two monthly files: keep the latest file's record
    df = df.sort_values("SourceMonth", kind="stable").drop_duplicates("ListingKey", keep="last")
    log.append(("dedupe ListingKey", len(df)))

    # drop rows whose target or main size field is clearly a data-entry error
    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    ppsf = df[TARGET] / df["LivingArea"]
    df = df[df["CloseDate"].notna()
            & df[TARGET].between(50_000, 30_000_000)
            & df["LivingArea"].between(300, 20_000)
            & ppsf.between(25, 5_000)]
    log.append(("drop bad price / living area", len(df)))

    # impossible values -> NaN (imputed later, fit on train only)
    df.loc[~df["BedroomsTotal"].between(1, 15), "BedroomsTotal"] = np.nan
    df.loc[~df["BathroomsTotalInteger"].between(1, 15), "BathroomsTotalInteger"] = np.nan
    df.loc[~df["YearBuilt"].between(1850, df["CloseDate"].dt.year + 2), "YearBuilt"] = np.nan
    df.loc[df["LotSizeSquareFeet"] <= 0, "LotSizeSquareFeet"] = np.nan
    df.loc[~df["GarageSpaces"].between(0, 20), "GarageSpaces"] = np.nan
    df.loc[~df["ParkingTotal"].between(0, 50), "ParkingTotal"] = np.nan
    df.loc[df["MainLevelBedrooms"] > df["BedroomsTotal"], "MainLevelBedrooms"] = np.nan
    outside_ca = ~(df["Latitude"].between(*CA_LAT) & df["Longitude"].between(*CA_LON))
    df.loc[outside_ca, ["Latitude", "Longitude"]] = np.nan

    # categoricals: tidy text, explicit "Missing" level
    df["PostalCode"] = df["PostalCode"].astype(str).str[:5]
    df["Levels"] = df["Levels"].str.split(",").str[0]
    for c in LOW_CARD + HIGH_CARD:
        df[c] = df[c].fillna("Missing").astype(str)

    # missing coordinates -> median of the same ZIP (uses no price info, so no leakage)
    for c in ["Latitude", "Longitude"]:
        df[c] = df[c].fillna(df.groupby("PostalCode")[c].transform("median"))

    # HOA dues on one monthly scale; no frequency given -> monthly, no fee listed -> no HOA
    factor = df["AssociationFeeFrequency"].map(HOA_PER_MONTH).fillna(1)
    df["AssociationFeeMonthly"] = (df["AssociationFee"] * factor).fillna(0)

    for c in BINARY:
        df[c] = df[c].astype(float)              # True/False/NaN -> 1/0/NaN

    floors = df["Flooring"].fillna("").str.get_dummies(sep=",")
    for t, col in zip(FLOORING, FLOORING_COLS):
        df[col] = floors[t] if t in floors else 0

    if verbose:
        for step, n in log:
            print(f"{step:<32}{n:>9,}")
    return df[["ListingKey", "CloseDate", TARGET] + FEATURES].reset_index(drop=True)


def time_split(df, test_month, train_months):
    """Test = `test_month`; train = the `train_months` calendar months right before it."""
    test_month = pd.Period(test_month, "M")
    month = df["CloseDate"].dt.to_period("M")
    train = df[(month >= test_month - train_months) & (month < test_month)]
    test = df[month == test_month]
    return train, test


def build_preprocessor():
    """Imputation + encoding + scaling. Fit it on the training window only."""
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                 SimpleImputer(strategy="median", add_indicator=True),
                                 StandardScaler()), SKEWED),
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), NUMERIC),
        ("binary", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), BINARY),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                 sparse_output=False), LOW_CARD),
        # cross-fitted mean log-price per City/ZIP/area/district, so the encoder doesn't memorize train rows
        ("target", make_pipeline(TargetEncoder(target_type="continuous", random_state=0),
                                 StandardScaler()), HIGH_CARD),
        ("flooring", "passthrough", FLOORING_COLS),
    ], verbose_feature_names_out=False)


def price_model(estimator):
    """preprocessor -> estimator, trained on log(ClosePrice) and predicting back in dollars."""
    return TransformedTargetRegressor(make_pipeline(build_preprocessor(), estimator),
                                      func=np.log, inverse_func=np.exp)
