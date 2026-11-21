"""Week 9: California home price app.  Run:  streamlit run app.py
(model.joblib comes from train_model.py)"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CA Home Price Estimator", page_icon="🏠")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


m = load()
zips = m["zip_info"]

st.title("🏠 California Home Price Estimator")
st.caption("Predicts the sale price of a single-family home from CRMLS sales. "
           f"LightGBM trained on {m['trained_on']}.")

col1, col2 = st.columns(2)
living = col1.number_input("Living area (sq ft)", 300, 20000, 1800, step=50)
lot = col2.number_input("Lot size (sq ft)", 0, 500000, 6000, step=250)
beds = col1.number_input("Bedrooms", 1, 15, 3)
baths = col2.number_input("Bathrooms", 1, 15, 2)
zip_code = st.selectbox("ZIP code", zips.index, index=list(zips.index).index("92126") if "92126" in zips.index else 0,
                        format_func=lambda z: f"{z} — {zips.loc[z, 'County']} County")

# one row, built exactly like train_model.features()
info = zips.loc[zip_code]
X = pd.DataFrame([{"LivingArea": np.log1p(living), "BedroomsTotal": beds, "BathroomsTotalInteger": baths,
                   "LotSizeSquareFeet": np.log1p(lot or m["medians"]["LotSizeSquareFeet"]),
                   "ZipLat": info["ZipLat"], "ZipLon": info["ZipLon"],
                   "ZipPrice": m["prices"].get(zip_code, m["overall"])}])
X["County"] = pd.Categorical([info["County"]], categories=m["counties"])

price = float(np.exp(m["model"].predict(X)[0]))
lo, hi = price * np.exp(m["range"][0]), price * np.exp(m["range"][1])

st.divider()
st.metric("Predicted price", f"${price:,.0f}")
st.write(f"**90% range:** ${lo:,.0f} – ${hi:,.0f}")
st.caption(f"About 9 in 10 homes sell inside this range. ${price / living:,.0f} per sq ft.")

with st.expander("How accurate is this?"):
    mt = m["metrics"]
    st.write(f"Tested on May 2026 sales the model never saw: R² **{mt['test_R2']:.3f}**, "
             f"typical error **{mt['median_pct_error']:.1%}**, and the 90% range contained the real price "
             f"**{mt['range_coverage']:.1%}** of the time. Least accurate for luxury homes, since views and "
             "finishes aren't in the data.")
