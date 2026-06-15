# app.py / app_ui.py
# 3-page Smart Crop Advisor: Page1=Top3, Page2=Yield+Profit, Page3=Rotation

import streamlit as st
import joblib, json, numpy as np, xgboost as xgb, pandas as pd
from pathlib import Path
import altair as alt
import difflib
from datetime import datetime
import io
import time

# NEW: rule-based filter for realistic crops
from rules_state_climate import apply_all_filters, get_allowed_crops

# -------------------------
# Paths & loaders
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS = BASE_DIR / "models"

#st.write("BASE_DIR =", BASE_DIR)
#st.write("MODELS =", MODELS)
#st.write("MODELS EXISTS =", MODELS.exists())


def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


# Core artifacts
ohe = safe_load(MODELS / "ohe.pkl")
scaler = safe_load(MODELS / "scaler.pkl")
le = safe_load(MODELS / "labelencoder_crop.pkl")

xgb_clf = safe_load(MODELS / "xgb_classifier.model")
rf_clf = safe_load(MODELS / "rf_classifier.pkl")
meta = safe_load(MODELS / "meta_classifier.pkl")

xgb_yield = safe_load(MODELS / "xgb_yield_reg.model")
rf_yield = safe_load(MODELS / "rf_yield_reg.pkl")

ohe_crop = safe_load(MODELS / "ohe_crop.pkl")
scaler_reg = safe_load(MODELS / "scaler_reg.pkl")

try:
    rotation = json.load(open(MODELS / "rotation_rules.json"))
except Exception:
    rotation = {}

try:
    crop_means = json.load(open(MODELS / "crop_means.json"))
except Exception:
    crop_means = {}

# Validate critical artifacts
critical = {
    "ohe": ohe,
    "scaler": scaler,
    "le": le,
    "xgb_clf": xgb_clf,
    "rf_clf": rf_clf,
    "meta": meta,
}
missing = [k for k, v in critical.items() if v is None]
if missing:
    st.error(f"Missing model artifacts: {missing}. Put them in ../models and restart.")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
DEFAULT_MARKET_PRICE_FOR_RECO = 20000.0  # used only on Page 1 to compute suitability


def build_feature_row(
    state, soil, rainfall, humidity, temp, pH, N, P, K, input_cost, market_price
):
    """Features for classifiers (no crop in features)."""
    NPK_sum = N + P + K
    N_ratio = N / (NPK_sum + 1e-6)
    P_ratio = P / (NPK_sum + 1e-6)
    K_ratio = K / (NPK_sum + 1e-6)
    cost_to_price = input_cost / (market_price + 1e-6)

    num_feats = np.array(
        [
            rainfall,
            humidity,
            temp,
            pH,
            N,
            P,
            K,
            NPK_sum,
            N_ratio,
            P_ratio,
            K_ratio,
            input_cost,
            market_price,
            cost_to_price,
        ]
    ).reshape(1, -1)

    cat_feats = ohe.transform([[soil, state]])
    xrow = np.hstack([num_feats, cat_feats])
    xrow_scaled = scaler.transform(xrow)
    return xrow_scaled


def build_feature_row_for_regressor(
    state,
    soil,
    crop,
    rainfall,
    humidity,
    temp,
    pH,
    N,
    P,
    K,
    input_cost,
    market_price,
):
    """Features for regressors: numeric + soil/state OHE + crop OHE."""
    if ohe is None or ohe_crop is None or scaler_reg is None:
        return None

    NPK_sum = N + P + K
    N_ratio = N / (NPK_sum + 1e-6)
    P_ratio = P / (NPK_sum + 1e-6)
    K_ratio = K / (NPK_sum + 1e-6)
    cost_to_price = input_cost / (market_price + 1e-6)

    num_feats = np.array(
        [
            rainfall,
            humidity,
            temp,
            pH,
            N,
            P,
            K,
            NPK_sum,
            N_ratio,
            P_ratio,
            K_ratio,
            input_cost,
            market_price,
            cost_to_price,
        ]
    ).reshape(1, -1)

    cat_feats = ohe.transform([[soil, state]])
    crop_ohe = ohe_crop.transform([[crop]])
    xrow = np.hstack([num_feats, cat_feats, crop_ohe])
    xrow_scaled = scaler_reg.transform(xrow)
    return xrow_scaled


def normalize_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


_norm_rotation = {normalize_name(k): v for k, v in rotation.items()}


def get_rotation_for_crop(scrop: str):
    scrop_norm = normalize_name(scrop)
    if scrop_norm in _norm_rotation:
        return _norm_rotation[scrop_norm], f"Exact match ({scrop})"

    candidates = list(_norm_rotation.keys())
    matches = difflib.get_close_matches(scrop_norm, candidates, n=1, cutoff=0.7)
    if matches:
        key = matches[0]
        return _norm_rotation[key], f"Fuzzy match: {scrop} → {key}"

    s_low = scrop.lower()
    cereals = {"rice", "wheat", "maize", "bajra", "sorghum", "millet"}
    legumes = {
        "chickpea",
        "gram",
        "pigeonpeas",
        "pigeonpea",
        "urd",
        "moong",
        "mungbean",
        "lentil",
        "groundnut",
        "soyabean",
        "soybean",
    }
    fruits = {
        "mango",
        "banana",
        "apple",
        "papaya",
        "grapes",
        "pomegranate",
        "watermelon",
    }

    if any(c in s_low for c in cereals):
        return (
            ["Legumes (Chickpea, Lentil, Groundnut)", "Mustard", "Soybean"],
            "Family fallback (cereal→legume)",
        )
    if any(c in s_low for c in legumes):
        return (
            ["Cereals (Wheat, Maize, Bajra)", "Fodder/Green Manure"],
            "Family fallback (legume→cereal)",
        )
    if any(c in s_low for c in fruits):
        return (
            ["Vegetable/cover crop", "Fodder/legume intercrop"],
            "Family fallback (fruit)",
        )

    return (
        ["Legume (for N-fixation)", "Short-duration vegetable", "Fodder/Green manure"],
        "Generic fallback",
    )


# -------------------------
# Theme
# -------------------------
LIGHT_CSS = """
:root { --bg: #ffffff; --text: #0b1220; }
.stApp { background-color: var(--bg) !important; color: var(--text) !important; }
"""
DARK_CSS = """
:root { --bg: #0f1720; --text: #e6eef6; }
.stApp { background-color: var(--bg) !important; color: var(--text) !important; }
"""


def apply_theme_css(theme: str):
    st.markdown(
        f"<style>{LIGHT_CSS if theme=='light' else DARK_CSS}</style>",
        unsafe_allow_html=True,
    )


# -------------------------
# App config & session
# -------------------------
st.set_page_config(page_title="Smart Crop Advisor", layout="wide")

defaults = {
    "theme": "dark",
    "top3_crops": None,
    "top3_scores": None,
    "last_input": None,
    "selected_crop": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

apply_theme_css(st.session_state.get("theme", "dark"))

# -------------------------
# Sidebar: navigation + theme + reset
# -------------------------
with st.sidebar:
    st.title("🌾 Smart Crop Advisor")
    page = st.radio(
        "Navigation",
        ["1️⃣ Crop Recommendation", "2️⃣ Yield & Profit", "3️⃣ Rotation Planner"],
        index=0,
    )

    st.markdown("---")
    theme_sel = st.radio(
        "Theme",
        ["dark", "light"],
        index=0 if st.session_state.theme == "dark" else 1,
    )
    if theme_sel != st.session_state.theme:
        st.session_state.theme = theme_sel
        apply_theme_css(theme_sel)

    st.markdown("---")

    def do_reset():
        theme_keep = st.session_state.get("theme", "dark")
        keep_keys = {"theme"}
        for key in list(st.session_state.keys()):
            if key in keep_keys:
                continue
            try:
                del st.session_state[key]
            except Exception:
                pass
        st.session_state.theme = theme_keep
        apply_theme_css(theme_keep)
        st.success("App reset (theme preserved).")

    if st.button("Reset App"):
        do_reset()

    

st.markdown(
    "<h1 style='margin:0'>🌱 Smart Crop Advisory System</h1>",
    unsafe_allow_html=True,
)
st.write(
    "B.Tech Final Year Project — Crop Recommendation, Yield & Profit Prediction, Rotation Planner"
)

# -------------------------
# PAGE 1 – Crop Recommendation
# -------------------------
if page.startswith("1️⃣"):
    st.header("Step 1 — Enter Field Conditions & Get Top-3 Crops")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("🧾 Field Inputs")

        with st.form("input_form_page1"):
            state = st.selectbox("State", list(ohe.categories_[1]))
            soil = st.selectbox("Soil Type", list(ohe.categories_[0]))

            rainfall = st.number_input("Season Rainfall (mm)", value=700.0, step=10.0)
            humidity = st.number_input("Humidity (%)", value=70.0, step=1.0)
            temp = st.number_input("Temperature (°C)", value=25.0, step=0.5)
            pH = st.number_input("Soil pH", value=6.8, step=0.1)

            N = st.number_input("Nitrogen (kg/ha)", value=60.0, step=1.0)
            P = st.number_input("Phosphorus (kg/ha)", value=30.0, step=1.0)
            K = st.number_input("Potassium (kg/ha)", value=30.0, step=1.0)

            input_cost = st.number_input(
                "Input Cost (₹/ha)", value=20000.0, step=500.0
            )

            submitted = st.form_submit_button("🔍 Get Top-3 Crops")

    with right:
        if submitted:
            # Store input WITHOUT market_price (decided later)
            st.session_state.last_input = {
                "state": state,
                "soil": soil,
                "rainfall": rainfall,
                "humidity": humidity,
                "temp": temp,
                "pH": pH,
                "N": N,
                "P": P,
                "K": K,
                "input_cost": input_cost,
                "market_price": None,  # will be filled on Page 2
            }

            # For classification we assume an average market price
            xrow_s = build_feature_row(
                state,
                soil,
                rainfall,
                humidity,
                temp,
                pH,
                N,
                P,
                K,
                input_cost,
                DEFAULT_MARKET_PRICE_FOR_RECO,
            )

            # Probabilities
            try:
                proba_xgb = xgb_clf.predict(xgb.DMatrix(xrow_s))[0]
            except Exception:
                proba_xgb = xgb_clf.predict(xrow_s)[0]
            proba_rf = rf_clf.predict_proba(xrow_s)[0]

            meta_input = (proba_xgb + proba_rf) / 2.0
            final_probs = meta.predict_proba(meta_input.reshape(1, -1))[0]

            # Raw top-3 from model
            top_idx = np.argsort(final_probs)[-3:][::-1]
            raw_top_crops = le.inverse_transform(top_idx)
            raw_top_scores = final_probs[top_idx]
            raw_scores_map = {c: float(s) for c, s in zip(raw_top_crops, raw_top_scores)}

            # 🔥 Apply state + climate filters
            filtered_top = apply_all_filters(
                predicted_crops=list(raw_top_crops),
                state=state,
                rainfall_mm=rainfall,
                temperature_c=temp,
                top_k=3,
            )

            # If after filters we have < 3 crops, top-up from allowed crops of that state
            if len(filtered_top) < 3:
                allowed = get_allowed_crops(state)
                for c in allowed:
                    if c not in filtered_top:
                        filtered_top.append(c)
                    if len(filtered_top) == 3:
                        break

            # Suitability scores:
            # - if crop came from model → use its prob
            # - if crop was added by rules → assign slightly lower pseudo scores
            filtered_scores = []
            if len(raw_top_scores) > 0:
                base_default = float(np.mean(raw_top_scores)) * 0.8
            else:
                base_default = 0.4
            step = 0.02
            extra_i = 0

            for c in filtered_top:
                if c in raw_scores_map:
                    filtered_scores.append(raw_scores_map[c])
                else:
                    pseudo = max(base_default - extra_i * step, 0.1)
                    filtered_scores.append(pseudo)
                    extra_i += 1

            st.session_state.top3_crops = filtered_top
            st.session_state.top3_scores = filtered_scores

        st.subheader("🌿 Top-3 Recommended Crops")

        top3 = st.session_state.get("top3_crops")
        top_scores = st.session_state.get("top3_scores")

        if top3:
            cols = st.columns(len(top3))
            for i, (c, s) in enumerate(zip(top3, top_scores)):
                with cols[i]:
                    st.markdown(f"#### {c}")
                    st.metric("Suitability", f"{s:.3f}")

            df_scores = pd.DataFrame({"crop": top3, "score": top_scores})
            chart = (
                alt.Chart(df_scores)
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", title="Suitability Score"),
                    y=alt.Y("crop:N", sort="-x", title=None),
                    tooltip=["crop", "score"],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)

            st.info(
                "✅ Next: Go to **Page 2 – Yield & Profit** to select one crop and enter market price."
            )
        else:
            st.info(
                "Fill inputs on the left and click 'Get Top-3 Crops' to see recommendations."
            )

# -------------------------
# PAGE 2 – Compare Top-3
# -------------------------
elif page.startswith("2️⃣"):
    st.header("Step 2 — Compare Yield, Revenue & Profit for Top-3 Crops")

    top3 = st.session_state.get("top3_crops")
    top_scores = st.session_state.get("top3_scores")
    last_inp = st.session_state.get("last_input")

    if not top3 or not last_inp:
        st.warning("Please go to **Page 1** first and generate Top-3 crops.")
    else:
        st.subheader("💰 Enter Market Price for Each Recommended Crop")

        cols = st.columns(len(top3))
        prices = []
        for i, crop in enumerate(top3):
            with cols[i]:
                price = st.number_input(
                    f"{crop} price (₹ per ton / 1000 kg)",
                    value=20000.0,
                    step=500.0,
                    key=f"price_{crop}"
                )
                prices.append(price)

        if st.button("📈 Predict Yield & Profit for All Top-3"):
            results = []
            for crop, price, score in zip(top3, prices, top_scores):
                xrow_reg = build_feature_row_for_regressor(
                    last_inp["state"], last_inp["soil"], crop,
                    last_inp["rainfall"], last_inp["humidity"], last_inp["temp"],
                    last_inp["pH"], last_inp["N"], last_inp["P"], last_inp["K"],
                    last_inp["input_cost"], price
                )

                if xrow_reg is None:
                    if crop in crop_means:
                        yhat = float(crop_means[crop])
                    else:
                        xrow_s = build_feature_row(
                            last_inp["state"], last_inp["soil"],
                            last_inp["rainfall"], last_inp["humidity"], last_inp["temp"],
                            last_inp["pH"], last_inp["N"], last_inp["P"], last_inp["K"],
                            last_inp["input_cost"], price
                        )
                        try:
                            y1 = xgb_yield.predict(xgb.DMatrix(xrow_s))[0]
                        except Exception:
                            y1 = xgb_yield.predict(xrow_s)[0]
                        y2 = rf_yield.predict(xrow_s.reshape(1, -1))[0]
                        yhat = float((y1 + y2) / 2.0)
                else:
                    try:
                        y1 = xgb_yield.predict(xgb.DMatrix(xrow_reg))[0]
                    except Exception:
                        y1 = xgb_yield.predict(xrow_reg)[0]
                    y2 = rf_yield.predict(xrow_reg.reshape(1, -1))[0]
                    yhat = float((y1 + y2) / 2.0)

                revenue = (yhat / 1000.0) * price
                profit = revenue - last_inp["input_cost"]

                results.append({
                    "crop": crop,
                    "yield": yhat,
                    "revenue": revenue,
                    "profit": profit,
                    "suit": float(score)
                })

            # --- numeric cards in columns ---
            st.subheader("📊 Detailed Metrics for Each Crop")
            cols = st.columns(len(results))
            for col, res in zip(cols, results):
                with col:
                    st.markdown(f"### {res['crop']}")
                    st.metric("Suitability", f"{res['suit']:.3f}")
                    st.metric("Yield (kg/ha)", f"{res['yield']:,.2f}")
                    st.metric("Revenue (₹/ha)", f"{res['revenue']:,.2f}")
                    st.metric("Profit (₹/ha)", f"{res['profit']:,.2f}")

            # --- grouped bar chart like your sketch ---
            chart_rows = []
            for res in results:
                crop = res["crop"]
                chart_rows.append({"crop": crop, "metric": "Suitability (%)", "value": res["suit"] * 100})
                chart_rows.append({"crop": crop, "metric": "Yield (kg/ha)", "value": res["yield"]})
                chart_rows.append({"crop": crop, "metric": "Revenue (₹/ha)", "value": res["revenue"]})
                chart_rows.append({"crop": crop, "metric": "Profit (₹/ha)", "value": res["profit"]})

            df_chart = pd.DataFrame(chart_rows)

            metric_order = ["Yield (kg/ha)", "Revenue (₹/ha)", "Profit (₹/ha)"]
            metric_colors = ["#fbbf24", "#22c55e", "#f97373"]  # Y, R, P

            chart = (
                alt.Chart(df_chart)
                .mark_bar()
                .encode(
                    x=alt.X("crop:N", title="Crop"),
                    xOffset="metric:N",
                    y=alt.Y("value:Q", title="Value"),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(domain=metric_order, range=metric_colors),
                        legend=alt.Legend(title="Metric")
                    ),
                    tooltip=["crop", "metric", "value"]
                )
                .properties(height=400)
            )

            st.subheader("📊 Comparison of Yield, Revenue & Profit")
            st.altair_chart(chart, use_container_width=True)

            # download results CSV
            res_df = pd.DataFrame([{
                "crop": r["crop"],
                "suitability": r["suit"],
                "yield_kg_per_ha": r["yield"],
                "revenue_INR_per_ha": r["revenue"],
                "profit_INR_per_ha": r["profit"]
            } for r in results])
            csv_io = io.StringIO()
            res_df.to_csv(csv_io, index=False)
            st.download_button(
                "⬇️ Download Top-3 Comparison as CSV",
                csv_io.getvalue(),
                file_name="top3_comparison.csv",
                mime="text/csv"
            )

        else:
            st.info("Enter market prices, then click **Predict Yield & Profit for All Top-3**.")

# -------------------------
# PAGE 3 – Rotation Planner
# -------------------------
elif page.startswith("3️⃣"):
    st.header("Step 3 — Crop Rotation Planner")

    top3 = st.session_state.get("top3_crops")
    last_selected_crop = st.session_state.get("selected_crop")

    if not top3:
        st.warning("Please go to **Page 1** first and generate Top-3 crops.")
    else:
        st.subheader("🌿 Choose Crop for Rotation Planning")

        default_idx = 0
        if last_selected_crop and last_selected_crop in top3:
            default_idx = top3.index(last_selected_crop)

        crop_for_rotation = st.selectbox(
            "Select the crop you plan to grow", top3, index=default_idx
        )

        rules, source = get_rotation_for_crop(crop_for_rotation)

        st.markdown(f"### 🔄 Rotation Suggestions for **{crop_for_rotation}**")
        st.write(f"**Rule source:** {source}")
        for r in rules:
            st.write(f"- {r}")

        st.info(
            "Rotation planning helps maintain soil fertility, break pest cycles, and improve long-term yield."
        )
