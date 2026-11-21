"""Week 9: California home price app.  Run:  streamlit run app.py
(model.joblib comes from train_model.py)"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CA Home Price Estimator", page_icon="🪶")

# naturalist board-game look: parchment, watercolor washes, serif type, card panels
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=EB+Garamond:wght@400;600&display=swap');
.stApp {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(143,178,160,.35), transparent 45%),
    radial-gradient(ellipse at 95% 15%, rgba(214,168,120,.30), transparent 40%),
    radial-gradient(ellipse at 50% 110%, rgba(150,170,200,.25), transparent 50%),
    #f3ead3;
}
header[data-testid="stHeader"] { background: transparent; }
html, body, .stApp, p, label, input, div[data-baseweb] { font-family: 'EB Garamond', Georgia, serif !important; }
h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; color: #3d3324; letter-spacing: .5px; }
.block-container { background: rgba(251,246,233,.85); border: 2px solid #b9a67e; border-radius: 18px;
  box-shadow: 0 6px 18px rgba(80,60,30,.15); padding: 2.2rem 2.4rem !important; margin-top: 2rem; }
.card { border: 2px solid #8c7a55; border-radius: 16px; overflow: hidden; background: #fbf6e9;
  box-shadow: 0 4px 12px rgba(80,60,30,.18); }
.card-head { background: linear-gradient(90deg, #5b8a72, #7aa58a); color: #fbf6e9; padding: .5rem 1.2rem;
  font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; font-weight: 700; letter-spacing: 1px; }
.card-body { padding: 1rem 1.2rem 1.2rem; }
.price { font-family: 'Cormorant Garamond', serif; font-size: 3rem; font-weight: 700; color: #3d3324; line-height: 1.1; }
.sub { color: #6e5f45; font-size: 1.05rem; }
.range { position: relative; height: 12px; margin: 1.1rem 0 .35rem; border-radius: 6px;
  background: linear-gradient(90deg, #c9b88f, #d6a878 50%, #c9b88f); }
.dot { position: absolute; top: -5px; width: 22px; height: 22px; border-radius: 50%; background: #5b8a72;
  border: 3px solid #fbf6e9; box-shadow: 0 1px 4px rgba(0,0,0,.3); transform: translateX(-50%); }
.ends { display: flex; justify-content: space-between; color: #6e5f45; font-size: 1rem; }
.chips span { display: inline-block; margin: .7rem .4rem 0 0; padding: .15rem .7rem; border-radius: 999px;
  background: #e8dcbc; border: 1px solid #b9a67e; font-size: .95rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("model.joblib")


m = load()
zips = m["zip_info"]

st.title("🪶 California Home Price Estimator")
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

pos = 100 * (np.log(price) - np.log(lo)) / (np.log(hi) - np.log(lo))   # where the estimate sits in the range
st.markdown(f"""
<div class="card">
  <div class="card-head">Estimated Sale Price · ZIP {zip_code}</div>
  <div class="card-body">
    <div class="price">${price:,.0f}</div>
    <div class="sub">{info['County']} County · ${price / living:,.0f} per sq ft</div>
    <div class="range"><div class="dot" style="left:{pos:.0f}%"></div></div>
    <div class="ends"><span>${lo:,.0f}</span><span>90% range</span><span>${hi:,.0f}</span></div>
    <div class="chips"><span>{living:,} sq ft</span><span>{beds} bed</span><span>{baths} bath</span><span>{lot:,} sq ft lot</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("About 9 in 10 homes sell inside the 90% range.")

with st.expander("How accurate is this?"):
    mt = m["metrics"]
    st.write(f"Tested on May 2026 sales the model never saw: R² **{mt['test_R2']:.3f}**, "
             f"typical error **{mt['median_pct_error']:.1%}**, and the 90% range contained the real price "
             f"**{mt['range_coverage']:.1%}** of the time. Least accurate for luxury homes, since views and "
             "finishes aren't in the data.")
