# import streamlit as st

# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path

# st.set_page_config(
#     page_title="House Price Predictor 🏠",
#     page_icon="🏠",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# LAC = 100_000


# # ─── Load Model ───────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading model…")
# def load_model():
#     base = Path(__file__).parent
#     pipeline = joblib.load(base / "best_pipeline.joblib")
#     with open(base / "model_metadata.json") as f:
#         meta = json.load(f)
#     return pipeline, meta


# try:
#     pipeline, meta = load_model()
# except FileNotFoundError as e:
#     st.error(
#         "⚠️ Model files not found. Place `best_pipeline.joblib` and `model_metadata.json` next to `app.py`."
#     )
#     st.stop()

# num_stats = meta["num_stats"]
# categories = meta["categories"]
# num_cols = meta["num_cols"]
# cat_cols = meta["cat_cols"]


# def s(col, key):
#     return num_stats.get(col, {}).get(key, 0)


# # ─── CSS ──────────────────────────────────────────────────────
# st.markdown(
#     """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
# * { font-family: 'Inter', sans-serif; }
# .stApp { background: #F0F4FF; }
 
# /* Hero */
# .hero {
#     background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
#     border-radius: 20px;
#     padding: 2.5rem 3rem;
#     color: white;
#     margin-bottom: 2rem;
#     box-shadow: 0 8px 32px rgba(59,130,246,0.25);
# }
# .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
# .hero p  { font-size: 1.05rem; opacity: 0.8; margin: 0.5rem 0 0; }
 
# /* Cards */
# .card {
#     background: white;
#     border-radius: 16px;
#     padding: 1.6rem 1.8rem;
#     margin-bottom: 1.2rem;
#     box-shadow: 0 2px 12px rgba(0,0,0,0.06);
#     border: 1px solid #E8EEFE;
# }
# .card-title {
#     font-size: 1rem;
#     font-weight: 700;
#     color: #1E3A8A;
#     margin-bottom: 1.1rem;
#     display: flex;
#     align-items: center;
#     gap: 8px;
# }
 
# /* Result card */
# .result-card {
#     background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
#     border-radius: 20px;
#     padding: 2.2rem 2rem;
#     color: white;
#     text-align: center;
#     box-shadow: 0 8px 32px rgba(59,130,246,0.3);
# }
# .result-label { font-size: 0.9rem; opacity: 0.75; margin-bottom: 0.5rem; font-weight: 500; }
# .result-price { font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
# .result-range { font-size: 0.85rem; opacity: 0.65; margin-top: 0.7rem; }
 
# /* KPI row */
# .kpi-row { display: flex; gap: 10px; margin-top: 1rem; }
# .kpi-box {
#     flex: 1;
#     background: #F0F4FF;
#     border-radius: 12px;
#     padding: 0.9rem 0.5rem;
#     text-align: center;
# }
# .kpi-val { font-size: 1.15rem; font-weight: 700; color: #1E3A8A; }
# .kpi-lbl { font-size: 0.72rem; color: #64748B; margin-top: 3px; font-weight: 500; }
 
# /* Empty state */
# .empty-state {
#     background: white;
#     border-radius: 20px;
#     padding: 3rem 2rem;
#     text-align: center;
#     border: 2px dashed #BFDBFE;
# }
# .empty-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
# .empty-text { color: #94A3B8; font-size: 1rem; line-height: 1.6; }
 
# /* Input summary rows */
# .summary-row {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     padding: 7px 0;
#     border-bottom: 1px solid #F1F5F9;
#     font-size: 0.9rem;
# }
# .summary-row:last-child { border-bottom: none; }
# .summary-key { color: #64748B; font-weight: 500; }
# .summary-val { color: #1E293B; font-weight: 600; }
 
# /* Tips box */
# .tips-box {
#     background: #EFF6FF;
#     border-radius: 12px;
#     padding: 1rem 1.2rem;
#     border-left: 4px solid #3B82F6;
#     font-size: 0.85rem;
#     color: #1E40AF;
#     line-height: 1.7;
#     margin-top: 1rem;
# }
 
# /* Predict button */
# div.stButton > button {
#     background: linear-gradient(135deg, #1E3A8A, #3B82F6) !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 12px !important;
#     padding: 0.75rem 2rem !important;
#     font-size: 1.05rem !important;
#     font-weight: 700 !important;
#     width: 100% !important;
#     letter-spacing: 0.3px !important;
#     box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
#     transition: all 0.2s !important;
# }
# div.stButton > button:hover {
#     transform: translateY(-1px) !important;
#     box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
# }
 
# /* Hide streamlit branding */
# #MainMenu { visibility: hidden; }
# footer    { visibility: hidden; }
# header    { visibility: hidden; }
 
# /* Widget labels */
# label { font-weight: 600 !important; color: #374151 !important; font-size: 0.88rem !important; }
# </style>
# """,
#     unsafe_allow_html=True,
# )

# # ─── Hero ─────────────────────────────────────────────────────
# st.markdown(
#     """
# <div class="hero">
#   <h1>🏠 House Price Predictor</h1>
#   <p>Enter your property details below and get an instant price estimate</p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# # ─── Layout ───────────────────────────────────────────────────
# left, right = st.columns([3, 2], gap="large")

# # ════════════════════════════════════
# # LEFT — INPUTS
# # ════════════════════════════════════
# with left:

#     # Card 1: Location
#     st.markdown(
#         '<div class="card"><div class="card-title">📍 Where is the property?</div>',
#         unsafe_allow_html=True,
#     )
#     c1, c2 = st.columns(2)
#     location = c1.selectbox(
#         "City", categories.get("location", ["mumbai"]), help="Select the city"
#     )
#     transaction = c2.selectbox(
#         "Are you buying or renting?",
#         categories.get("Transaction", ["Resale"]),
#         help="New Property = brand new · Resale = second-hand · Rent/Lease = rental",
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Card 2: Size
#     st.markdown(
#         '<div class="card"><div class="card-title">📐 Property Size</div>',
#         unsafe_allow_html=True,
#     )
#     c3, c4 = st.columns(2)
#     carpet_area = c3.number_input(
#         "Carpet Area (sqft)",
#         min_value=int(s("Carpet Area", "min")),
#         max_value=int(s("Carpet Area", "max")),
#         value=int(s("Carpet Area", "median")),
#         step=50,
#         help="The actual usable area inside the apartment",
#     )
#     super_area = c4.number_input(
#         "Super Area (sqft)",
#         min_value=int(s("Super Area", "min")),
#         max_value=int(s("Super Area", "max")),
#         value=int(s("Super Area", "median")),
#         step=50,
#         help="Total area including walls, corridors and common areas",
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Card 3: Rooms
#     st.markdown(
#         '<div class="card"><div class="card-title">�️ Rooms & Layout</div>',
#         unsafe_allow_html=True,
#     )
#     c5, c6, c7 = st.columns(3)
#     bhk = c5.slider(
#         "Bedrooms (BHK)",
#         int(s("BHK", "min")),
#         int(s("BHK", "max")),
#         int(s("BHK", "median")),
#     )
#     bathroom = c6.slider(
#         "Bathrooms",
#         int(s("Bathroom", "min")),
#         int(s("Bathroom", "max")),
#         int(s("Bathroom", "median")),
#     )
#     balcony = c7.slider(
#         "Balconies",
#         int(s("Balcony", "min")),
#         int(s("Balcony", "max")),
#         int(s("Balcony", "median")),
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Card 4: Building
#     st.markdown(
#         '<div class="card"><div class="card-title">🏢 Floor & Building</div>',
#         unsafe_allow_html=True,
#     )
#     c8, c9 = st.columns(2)
#     floor_num = c8.number_input(
#         "Which floor is the apartment on?",
#         min_value=0,
#         max_value=int(s("Total_Floor", "max")),
#         value=int(s("Floor_num", "median")),
#         help="Ground floor = 0",
#     )
#     total_floor = c9.number_input(
#         "Total floors in the building",
#         min_value=1,
#         max_value=int(s("Total_Floor", "max")),
#         value=int(s("Total_Floor", "median")),
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Card 5: Extra details
#     st.markdown(
#         '<div class="card"><div class="card-title">✨ Extra Details</div>',
#         unsafe_allow_html=True,
#     )
#     c10, c11 = st.columns(2)
#     furnishing = c10.selectbox(
#         "Furnishing",
#         categories.get("Furnishing", ["Semi-Furnished"]),
#         help="Furnished = has furniture · Semi = partial · Unfurnished = empty",
#     )
#     car_parking = c11.selectbox(
#         "Parking",
#         categories.get("Car Parking", ["None"]),
#         help="Type of parking available with the property",
#     )
#     c12, c13 = st.columns(2)
#     facing = c12.selectbox(
#         "Facing Direction",
#         categories.get("facing", ["East"]),
#         help="Which direction does the main entrance face?",
#     )
#     ownership = c13.selectbox(
#         "Ownership Type",
#         categories.get("Ownership", ["Freehold"]),
#         help="Freehold = you own the land too · Leasehold = land is rented",
#     )
#     overlooking = st.selectbox(
#         "What does the property overlook?",
#         categories.get("overlooking", ["Garden/Park"]),
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Predict button
#     predict_btn = st.button("🔍 Get Price Estimate", use_container_width=True)

# # ─── Build input dataframe ────────────────────────────────────
# floor_ratio = floor_num / max(total_floor, 1)
# bath_per_bhk = bathroom / max(bhk, 1)
# price_per_sqft = float(s("price_per_sqft", "median"))


# def make_input():
#     row = {
#         "Carpet Area": float(carpet_area),
#         "Super Area": float(super_area),
#         "BHK": float(bhk),
#         "Bathroom": float(bathroom),
#         "Balcony": float(balcony),
#         "Floor_num": float(floor_num),
#         "Total_Floor": float(total_floor),
#         "price_per_sqft": price_per_sqft,
#         "floor_ratio": floor_ratio,
#         "bath_per_bhk": bath_per_bhk,
#         "location": location,
#         "Transaction": transaction,
#         "Furnishing": furnishing,
#         "facing": facing,
#         "overlooking": overlooking,
#         "Car Parking": car_parking,
#         "Ownership": ownership,
#     }
#     all_cols = num_cols + cat_cols
#     return pd.DataFrame([{k: v for k, v in row.items() if k in all_cols}])


# # ════════════════════════════════════
# # RIGHT — RESULT
# # ════════════════════════════════════
# with right:

#     # Run prediction
#     if predict_btn:
#         with st.spinner("Calculating estimate…"):
#             pred_log = pipeline.predict(make_input())[0]
#             pred_lac = np.expm1(pred_log)
#         st.session_state.update(
#             pred=pred_lac, lo=pred_lac * 0.84, hi=pred_lac * 1.16, done=True
#         )

#     # ── Show result or empty state ────────────────────────────
#     if st.session_state.get("done"):
#         p = st.session_state["pred"]
#         lo = st.session_state["lo"]
#         hi = st.session_state["hi"]

#         def fmt(v):
#             return f"₹{v/100:.2f} Cr" if v >= 100 else f"₹{v:.1f} Lac"

#         st.markdown(
#             f"""
#         <div class="result-card">
#           <div class="result-label">Estimated Price</div>
#           <div class="result-price">{fmt(p)}</div>
#           <div class="result-range">Likely range: {fmt(lo)} — {fmt(hi)}</div>
#         </div>
#         <div class="kpi-row">
#           <div class="kpi-box">
#             <div class="kpi-val">₹{p*LAC/max(carpet_area,1):,.0f}</div>
#             <div class="kpi-lbl">Price per sqft</div>
#           </div>
#           <div class="kpi-box">
#             <div class="kpi-val">{fmt(p)}</div>
#             <div class="kpi-lbl">Total Price</div>
#           </div>
#           <div class="kpi-box">
#             <div class="kpi-val">{fmt(p/max(bhk,1))}</div>
#             <div class="kpi-lbl">Per Bedroom</div>
#           </div>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#         # Tips box
#         st.markdown(
#             """
#         <div class="tips-box">
#           💡 <b>How to read this:</b><br>
#           The estimate is based on 128,000 real listings across India.
#           The likely range shows the minimum and maximum the property
#           could realistically be priced at, based on similar properties.
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     else:
#         st.markdown(
#             """
#         <div class="empty-state">
#           <div class="empty-icon">🏠</div>
#           <div class="empty-text">
#             Fill in the property details<br>on the left and click<br>
#             <b>Get Price Estimate</b>
#           </div>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     # ── Input summary card ────────────────────────────────────
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown(
#         '<div class="card"><div class="card-title">📋 Your Property Summary</div>',
#         unsafe_allow_html=True,
#     )

#     summary = [
#         ("City", location),
#         ("Type", transaction),
#         ("Size", f"{carpet_area:,} sqft"),
#         ("Bedrooms", f"{bhk} BHK"),
#         ("Bathrooms", bathroom),
#         ("Floor", f"{int(floor_num)} of {int(total_floor)}"),
#         ("Furnishing", furnishing),
#         ("Parking", car_parking),
#         ("Facing", facing),
#     ]
#     rows_html = ""
#     for k, v in summary:
#         rows_html += f"""
#         <div class="summary-row">
#           <span class="summary-key">{k}</span>
#           <span class="summary-val">{v}</span>
#         </div>"""
#     st.markdown(rows_html, unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)


import streamlit as st
import json
import numpy as np
import pandas as pd
from pathlib import Path
 
st.set_page_config(
    page_title="House Price Predictor 🏠",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
LAC = 100_000
 
# ─── Pure-numpy model loader (no sklearn needed at runtime) ───
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    base = Path(__file__).parent
 
    with open(base / "model_data.json") as f:
        md = json.load(f)
    with open(base / "model_metadata.json") as f:
        meta = json.load(f)
 
    # Rebuild arrays
    md['support_vectors'] = np.array(md['support_vectors'])
    md['dual_coef']       = np.array(md['dual_coef'])
    md['intercept']       = np.array(md['intercept'])
    md['pca_components']  = np.array(md['pca_components'])
    md['pca_mean']        = np.array(md['pca_mean'])
    md['scaler_mean']     = np.array(md['scaler_mean'])
    md['scaler_scale']    = np.array(md['scaler_scale'])
    md['imputer_stats']   = np.array(md['imputer_stats'])
    md['ohe_imputer']     = np.array(md['ohe_imputer'])
    md['ohe_categories']  = [np.array(c) for c in md['ohe_categories']]
 
    return md, meta
 
try:
    md, meta = load_model()
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}\n\nPlace `model_data.json` and `model_metadata.json` next to `app.py`.")
    st.stop()
 
num_stats  = meta["num_stats"]
categories = meta["categories"]
num_cols   = meta["num_cols"]
cat_cols   = meta["cat_cols"]
 
def s(col, key):
    return num_stats.get(col, {}).get(key, 0)
 
# ─── Pure numpy predict ────────────────────────────────────────
def predict_price(row_dict):
    # 1. Numeric: impute → scale
    num_vals = []
    for i, col in enumerate(num_cols):
        v = row_dict.get(col, np.nan)
        if np.isnan(float(v)):
            v = md['imputer_stats'][i]
        num_vals.append(float(v))
    num_arr = (np.array(num_vals) - md['scaler_mean']) / md['scaler_scale']
 
    # 2. Categorical: impute → OHE
    cat_enc = []
    for i, col in enumerate(cat_cols):
        v = row_dict.get(col, None)
        cats = md['ohe_categories'][i]
        if v is None or v not in cats:
            v = md['ohe_imputer'][i]
        one_hot = (cats == v).astype(float)
        cat_enc.extend(one_hot.tolist())
 
    # 3. Combine
    X = np.concatenate([num_arr, np.array(cat_enc)]).reshape(1, -1)
 
    # 4. PCA transform
    X_pca = (X - md['pca_mean']) @ md['pca_components'].T
 
    # 5. RBF SVR predict
    sv   = md['support_vectors']
    dc   = md['dual_coef']
    gamma = md['params'].get('svr__gamma', 'scale')
 
    if gamma == 'scale':
        gamma_val = 1.0 / (X_pca.shape[1] * X_pca.var()) if X_pca.var() > 0 else 1.0
    elif gamma == 'auto':
        gamma_val = 1.0 / X_pca.shape[1]
    else:
        gamma_val = float(gamma)
 
    # RBF kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
    diff      = X_pca - sv                          # (n_sv, n_features)
    sq_dist   = np.sum(diff ** 2, axis=1)           # (n_sv,)
    K         = np.exp(-gamma_val * sq_dist)        # (n_sv,)
    pred      = float(dc @ K) + float(md['intercept'][0])
 
    return np.expm1(pred)   # back-transform from log scale
 
# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #F0F4FF; }
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    border-radius: 20px; padding: 2.5rem 3rem;
    color: white; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(59,130,246,0.25);
}
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; opacity: 0.8; margin: 0.5rem 0 0; }
.card {
    background: white; border-radius: 16px;
    padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #E8EEFE;
}
.card-title {
    font-size: 1rem; font-weight: 700; color: #1E3A8A;
    margin-bottom: 1.1rem; display: flex; align-items: center; gap: 8px;
}
.result-card {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    border-radius: 20px; padding: 2.2rem 2rem;
    color: white; text-align: center;
    box-shadow: 0 8px 32px rgba(59,130,246,0.3);
}
.result-label { font-size: 0.9rem; opacity: 0.75; margin-bottom: 0.5rem; font-weight: 500; }
.result-price { font-size: 3.2rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
.result-range { font-size: 0.85rem; opacity: 0.65; margin-top: 0.7rem; }
.kpi-row { display: flex; gap: 10px; margin-top: 1rem; }
.kpi-box { flex: 1; background: #F0F4FF; border-radius: 12px; padding: 0.9rem 0.5rem; text-align: center; }
.kpi-val { font-size: 1.15rem; font-weight: 700; color: #1E3A8A; }
.kpi-lbl { font-size: 0.72rem; color: #64748B; margin-top: 3px; font-weight: 500; }
.empty-state {
    background: white; border-radius: 20px; padding: 3rem 2rem;
    text-align: center; border: 2px dashed #BFDBFE;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.empty-text { color: #94A3B8; font-size: 1rem; line-height: 1.6; }
.summary-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.9rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #64748B; font-weight: 500; }
.summary-val { color: #1E293B; font-weight: 600; }
.tips-box {
    background: #EFF6FF; border-radius: 12px; padding: 1rem 1.2rem;
    border-left: 4px solid #3B82F6; font-size: 0.85rem;
    color: #1E40AF; line-height: 1.7; margin-top: 1rem;
}
div.stButton > button {
    background: linear-gradient(135deg, #1E3A8A, #3B82F6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important; font-weight: 700 !important;
    width: 100% !important; box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
}
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
label { font-weight: 600 !important; color: #374151 !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)
 
# ─── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏠 House Price Predictor</h1>
  <p>Enter your property details below and get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)
 
left, right = st.columns([3, 2], gap="large")
 
# ════════════ LEFT — INPUTS ════════════
with left:
    st.markdown('<div class="card"><div class="card-title">📍 Where is the property?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    location    = c1.selectbox("City", categories.get("location", ["mumbai"]))
    transaction = c2.selectbox("Are you buying or renting?", categories.get("Transaction", ["Resale"]),
                               help="New Property = brand new · Resale = second-hand · Rent/Lease = rental")
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="card"><div class="card-title">📐 Property Size</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    carpet_area = c3.number_input("Carpet Area (sqft)", min_value=int(s("Carpet Area","min")),
                                   max_value=int(s("Carpet Area","max")), value=int(s("Carpet Area","median")),
                                   step=50, help="The actual usable area inside the apartment")
    super_area  = c4.number_input("Super Area (sqft)",  min_value=int(s("Super Area","min")),
                                   max_value=int(s("Super Area","max")), value=int(s("Super Area","median")),
                                   step=50, help="Total area including walls and common areas")
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="card"><div class="card-title">🛏️ Rooms & Layout</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    bhk      = c5.slider("Bedrooms (BHK)", int(s("BHK","min")),      int(s("BHK","max")),      int(s("BHK","median")))
    bathroom = c6.slider("Bathrooms",      int(s("Bathroom","min")), int(s("Bathroom","max")), int(s("Bathroom","median")))
    balcony  = c7.slider("Balconies",      int(s("Balcony","min")),  int(s("Balcony","max")),  int(s("Balcony","median")))
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="card"><div class="card-title">🏢 Floor & Building</div>', unsafe_allow_html=True)
    c8, c9 = st.columns(2)
    floor_num   = c8.number_input("Which floor is the apartment on?", min_value=0,
                                   max_value=int(s("Total_Floor","max")), value=int(s("Floor_num","median")),
                                   help="Ground floor = 0")
    total_floor = c9.number_input("Total floors in the building", min_value=1,
                                   max_value=int(s("Total_Floor","max")), value=int(s("Total_Floor","median")))
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="card"><div class="card-title">✨ Extra Details</div>', unsafe_allow_html=True)
    c10, c11 = st.columns(2)
    furnishing  = c10.selectbox("Furnishing",  categories.get("Furnishing",  ["Semi-Furnished"]),
                                 help="Furnished = has furniture · Semi = partial · Unfurnished = empty")
    car_parking = c11.selectbox("Parking",     categories.get("Car Parking", ["None"]))
    c12, c13 = st.columns(2)
    facing      = c12.selectbox("Facing Direction", categories.get("facing",     ["East"]),
                                 help="Which direction does the main entrance face?")
    ownership   = c13.selectbox("Ownership Type",   categories.get("Ownership",  ["Freehold"]),
                                 help="Freehold = you own the land too · Leasehold = land is rented")
    overlooking = st.selectbox("What does the property overlook?", categories.get("overlooking", ["Garden/Park"]))
    st.markdown('</div>', unsafe_allow_html=True)
 
    predict_btn = st.button("🔍 Get Price Estimate", use_container_width=True)
 
# ─── Compute engineered features ──────────────────────────────
floor_ratio    = floor_num / max(total_floor, 1)
bath_per_bhk   = bathroom  / max(bhk, 1)
price_per_sqft = float(s("price_per_sqft", "median"))
 
def make_row():
    return {
        "Carpet Area": float(carpet_area), "Super Area": float(super_area),
        "BHK": float(bhk), "Bathroom": float(bathroom), "Balcony": float(balcony),
        "Floor_num": float(floor_num), "Total_Floor": float(total_floor),
        "price_per_sqft": price_per_sqft, "floor_ratio": floor_ratio,
        "bath_per_bhk": bath_per_bhk,
        "location": location, "Transaction": transaction, "Furnishing": furnishing,
        "facing": facing, "overlooking": overlooking,
        "Car Parking": car_parking, "Ownership": ownership,
    }
 
# ════════════ RIGHT — RESULT ════════════
with right:
    if predict_btn:
        with st.spinner("Calculating estimate…"):
            pred_lac = predict_price(make_row())
        st.session_state.update(pred=pred_lac, lo=pred_lac*0.84, hi=pred_lac*1.16, done=True)
 
    if st.session_state.get("done"):
        p, lo, hi = st.session_state["pred"], st.session_state["lo"], st.session_state["hi"]
        fmt = lambda v: f"₹{v/100:.2f} Cr" if v >= 100 else f"₹{v:.1f} Lac"
 
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Estimated Price</div>
          <div class="result-price">{fmt(p)}</div>
          <div class="result-range">Likely range: {fmt(lo)} — {fmt(hi)}</div>
        </div>
        <div class="kpi-row">
          <div class="kpi-box">
            <div class="kpi-val">₹{p*LAC/max(carpet_area,1):,.0f}</div>
            <div class="kpi-lbl">Price per sqft</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val">{fmt(p)}</div>
            <div class="kpi-lbl">Total Price</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val">{fmt(p/max(bhk,1))}</div>
            <div class="kpi-lbl">Per Bedroom</div>
          </div>
        </div>
        <div class="tips-box">
          💡 <b>How to read this:</b><br>
          Based on 128,000 real Indian property listings.
          The likely range shows what similar properties are realistically priced at.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🏠</div>
          <div class="empty-text">
            Fill in the property details on the left<br>and click <b>Get Price Estimate</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">📋 Your Property Summary</div>', unsafe_allow_html=True)
    rows_html = ""
    for k, v in [("City", location), ("Type", transaction), ("Size", f"{carpet_area:,} sqft"),
                 ("Bedrooms", f"{bhk} BHK"), ("Bathrooms", bathroom),
                 ("Floor", f"{int(floor_num)} of {int(total_floor)}"),
                 ("Furnishing", furnishing), ("Parking", car_parking), ("Facing", facing)]:
        rows_html += f'<div class="summary-row"><span class="summary-key">{k}</span><span class="summary-val">{v}</span></div>'
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("---")
st.caption("Predictions are estimates only · Not financial advice")
 