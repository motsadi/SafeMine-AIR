
import io
import os
import math
import json
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

APP_NAME = "SafeMine AIR™"
DATA_DIR = Path("data")
ASSETS_DIR = Path("assets")

DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "synthetic_gas_decay.csv"
MODEL_PATH = DATA_DIR / "model.json"  # store metadata; model pickled separately
MODEL_PKL = DATA_DIR / "model.pkl"

try:
    import joblib
except Exception:
    joblib = None


# -----------------------------
# Branding + page config
# -----------------------------
st.set_page_config(
    page_title=f"{APP_NAME} – Re-entry Prediction",
    page_icon="🛡️",
    layout="wide",
)

# Small CSS for a clean, professional feel
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .safemine-hero {
        padding: 1.0rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(3,102,214,0.10), rgba(76,175,80,0.10));
        border: 1px solid rgba(0,0,0,0.06);
      }
      .metric-card {
        padding: 0.9rem 1.0rem;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        background: rgba(255,255,255,0.6);
      }
      .small-muted { color: rgba(0,0,0,0.55); font-size: 0.92rem; }
      .tag {
        display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.10); background: rgba(0,0,0,0.03);
        font-size: 0.82rem; margin-right: 0.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Branding ---
LOGO_PATH = ASSETS_DIR / "navon_labs_logo.png"
SIDEBAR_LOGO_MAX_WIDTH = 180

def render_branding():
    # Sidebar logo + product title
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=SIDEBAR_LOGO_MAX_WIDTH)
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption("by Navon Labs")


# -----------------------------
# Text content (summarised)
# -----------------------------
SYSTEM_OVERVIEW = (
    "SafeMine AIR™ is a predictive software system designed to estimate safe re-entry times "
    "in underground mines after blasting by modelling the decay of toxic gases and evaluating ventilation performance. "
    "It blends machine learning for gas decay forecasting, ventilation physics for airflow/dilution, "
    "and decision-support analytics for ventilation planning. It requires no integration with mine planning systems."
)

PROBLEM_SUMMARY = (
    "Underground re-entry decisions often rely on conservative time-based guidelines, irregular manual gas measurements, "
    "and operator judgement. This creates safety risks (gas exposure), lost productivity (avoidable downtime), "
    "operational uncertainty (no forecasting), and underutilised ventilation and gas logs."
)

# -----------------------------
# Synthetic data generation
# -----------------------------
GASES = ["CO", "NOx", "SO2"]

def _alpha_function(Q, V, duct_len, duct_diam, aux_fans, temp_c, humidity, gas):
    """
    Hidden 'true' correction factor that a mine-specific model would learn.
    alpha > 0; higher alpha => faster clearance than baseline Q/V
    """
    # Resistance proxy: higher length and smaller diameter -> lower alpha
    resistance = (duct_len / 200.0) * (1.0 / max(duct_diam, 0.15))
    aux_gain = 0.06 * aux_fans
    env = 0.003 * (temp_c - 25.0) - 0.0015 * (humidity - 50.0)

    gas_factor = {"CO": 1.00, "NOx": 0.92, "SO2": 0.88}[gas]
    # nonlinear coupling: airflow helpful but diminishing returns
    airflow_term = 0.55 + 0.65 * (1.0 - np.exp(-Q / 45.0))

    alpha = gas_factor * airflow_term * (1.0 + aux_gain) * (1.0 + env) * np.exp(-0.18 * resistance)
    # clamp into reasonable band
    return float(np.clip(alpha, 0.25, 1.70))

def generate_synthetic_dataset(n_events=380, points_per_event=14, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2025, 1, 1, 8, 0, 0)

    for e in range(n_events):
        gas = rng.choice(GASES, p=[0.45, 0.35, 0.20])
        Q = float(rng.uniform(12, 120))  # m^3/s
        V = float(rng.uniform(800, 12000))  # m^3
        duct_len = float(rng.uniform(30, 450))  # m
        duct_diam = float(rng.uniform(0.25, 1.6))  # m
        aux_fans = int(rng.integers(0, 5))
        temp_c = float(rng.uniform(18, 38))
        humidity = float(rng.uniform(25, 85))
        C0 = float(rng.uniform(40, 800))  # ppm (synthetic starting concentration)

        # true alpha + event noise
        alpha_true = _alpha_function(Q, V, duct_len, duct_diam, aux_fans, temp_c, humidity, gas)
        alpha_true = float(np.clip(alpha_true + rng.normal(0, 0.06), 0.22, 1.85))

        # sample time points (minutes)
        t_max = float(rng.uniform(40, 180))
        ts = np.sort(rng.uniform(5, t_max, size=points_per_event))
        # convert minutes to seconds for physics
        for t_min in ts:
            t_sec = t_min * 60.0
            # baseline exponential with alpha
            Ct = C0 * math.exp(- (Q / V) * alpha_true * t_sec)
            # sensor noise + occasional missingness
            noise = rng.normal(0, 0.04)  # multiplicative noise in log-space
            Ct_noisy = Ct * math.exp(noise)

            # clamp and simulate floor
            Ct_noisy = float(max(Ct_noisy, 0.05))
            # add row
            rows.append({
                "event_id": e,
                "timestamp": (start + pd.to_timedelta(e * 6, unit="h")).isoformat(),
                "gas": gas,
                "Q_m3s": Q,
                "V_m3": V,
                "duct_length_m": duct_len,
                "duct_diameter_m": duct_diam,
                "aux_fans": aux_fans,
                "temp_C": temp_c,
                "humidity_pct": humidity,
                "C0_ppm": C0,
                "t_min": float(t_min),
                "Ct_ppm": float(Ct_noisy),
            })

    df = pd.DataFrame(rows)

    # Introduce some missing values realistically
    rng2 = np.random.default_rng(seed + 7)
    for col, p in [("temp_C", 0.02), ("humidity_pct", 0.02), ("duct_diameter_m", 0.01), ("Ct_ppm", 0.01)]:
        mask = rng2.uniform(0, 1, size=len(df)) < p
        df.loc[mask, col] = np.nan

    return df

def ensure_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = generate_synthetic_dataset()
    df.to_csv(DATA_PATH, index=False)
    return df


# -----------------------------
# Modeling
# -----------------------------
FEATURE_COLS = [
    "gas", "Q_m3s", "V_m3", "duct_length_m", "duct_diameter_m",
    "aux_fans", "temp_C", "humidity_pct", "C0_ppm"
]

def compute_alpha_from_row(row):
    """
    Estimate alpha from a single observation:
    Ct = C0 * exp(-(Q/V)*alpha*t)
    -> alpha = -(V/Q) * ln(Ct/C0) / t
    """
    try:
        Ct = float(row["Ct_ppm"])
        C0 = float(row["C0_ppm"])
        Q = float(row["Q_m3s"])
        V = float(row["V_m3"])
        t_sec = float(row["t_min"]) * 60.0
        if any([Ct <= 0, C0 <= 0, Q <= 0, V <= 0, t_sec <= 0]):
            return np.nan
        ratio = Ct / C0
        if ratio <= 0 or ratio >= 1.0:
            # ratio>=1 can happen with noise; handle as near 1
            ratio = min(max(ratio, 1e-6), 0.999999)
        alpha = -(V / Q) * math.log(ratio) / t_sec
        if not np.isfinite(alpha):
            return np.nan
        return float(np.clip(alpha, 0.05, 3.0))
    except Exception:
        return np.nan

def build_model(model_name="RandomForest"):
    numeric_features = [c for c in FEATURE_COLS if c != "gas"]
    categorical_features = ["gas"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    if model_name == "RandomForest":
        reg = RandomForestRegressor(
            n_estimators=350, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_leaf=2
        )
    elif model_name == "GradientBoosting":
        reg = GradientBoostingRegressor(random_state=42)
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(96, 48), random_state=42,
            max_iter=500, early_stopping=True
        )

    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
    return pipe

@st.cache_data(show_spinner=False)
def load_training_frame():
    df = ensure_dataset()
    df["alpha_est"] = df.apply(compute_alpha_from_row, axis=1)
    df = df.dropna(subset=["alpha_est"])
    return df

def train_and_save(model_name="RandomForest", test_size=0.2):
    df = load_training_frame()
    X = df[FEATURE_COLS].copy()
    y = df["alpha_est"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = build_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(mean_squared_error(y_test, preds) ** 0.5),
        "R2": float(r2_score(y_test, preds)),
        "n_rows": int(len(df)),
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    if joblib is None:
        raise RuntimeError("joblib is required to save/load the trained model. Install joblib.")
    joblib.dump(model, MODEL_PKL)
    MODEL_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics

def load_model():
    if joblib is None:
        return None, None
    if not MODEL_PKL.exists():
        return None, None
    model = joblib.load(MODEL_PKL)
    meta = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else None
    return model, meta

def hybrid_decay_curve(C0, Q, V, alpha, t_minutes):
    t_sec = np.array(t_minutes, dtype=float) * 60.0
    return C0 * np.exp(- (Q / V) * alpha * t_sec)

def time_to_threshold(C0, Q, V, alpha, threshold_ppm):
    threshold_ppm = max(float(threshold_ppm), 1e-6)
    if threshold_ppm >= C0:
        return 0.0
    # C(t) = C0 exp(-(Q/V) alpha t) => t = -(V/(Q alpha)) ln(th/C0)
    t_sec = - (V / (Q * alpha)) * math.log(threshold_ppm / C0)
    return max(t_sec / 60.0, 0.0)


# -----------------------------
# PDF Reporting
# -----------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def generate_pdf_report(payload: dict, curve_png: bytes) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Header
    c.setFillColor(colors.HexColor("#0B3D91"))
    c.rect(0, h - 2.2*cm, w, 2.2*cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.3*cm, h - 1.35*cm, "SafeMine AIR™ – Re-entry Prediction Report")
    c.setFont("Helvetica", 9)
    c.drawString(1.3*cm, h - 1.85*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y = h - 3.0*cm
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3*cm, y, "Summary")
    y -= 0.6*cm

    c.setFont("Helvetica", 10)
    summary_lines = [
        f"Gas: {payload['gas']}   |   Initial concentration C0: {payload['C0_ppm']:.1f} ppm   |   Threshold: {payload['threshold_ppm']:.1f} ppm",
        f"Airflow Q: {payload['Q_m3s']:.2f} m³/s   |   Volume V: {payload['V_m3']:.0f} m³   |   Aux fans: {payload['aux_fans']}",
        f"Predicted correction factor (alpha): {payload['alpha']:.3f}",
        f"Predicted safe re-entry time: {payload['t_safe_min']:.1f} minutes",
        f"Risk classification: {payload['risk_label']}",
    ]
    for line in summary_lines:
        c.drawString(1.3*cm, y, line)
        y -= 0.45*cm

    y -= 0.2*cm
    c.setStrokeColor(colors.HexColor("#D0D7DE"))
    c.line(1.3*cm, y, w-1.3*cm, y)
    y -= 0.6*cm

    # Parameter table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3*cm, y, "Inputs")
    y -= 0.6*cm

    table_data = [
        ["Parameter", "Value"],
        ["Gas", payload["gas"]],
        ["C0 (ppm)", f"{payload['C0_ppm']:.1f}"],
        ["Threshold (ppm)", f"{payload['threshold_ppm']:.1f}"],
        ["Q airflow (m³/s)", f"{payload['Q_m3s']:.2f}"],
        ["V volume (m³)", f"{payload['V_m3']:.0f}"],
        ["Duct length (m)", f"{payload['duct_length_m']:.0f}"],
        ["Duct diameter (m)", f"{payload['duct_diameter_m']:.2f}"],
        ["Aux fans", str(payload["aux_fans"])],
        ["Temperature (°C)", f"{payload['temp_C']:.1f}"],
        ["Humidity (%)", f"{payload['humidity_pct']:.1f}"],
    ]
    tbl = Table(table_data, colWidths=[6.0*cm, 9.0*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F6F8FA")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111827")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D0D7DE")),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FBFCFD")]),
    ]))
    tbl.wrapOn(c, w, h)
    tbl_h = 0.45*cm * len(table_data)
    tbl.drawOn(c, 1.3*cm, y - tbl_h + 0.2*cm)
    y -= (tbl_h + 0.8*cm)

    # Curve plot image
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3*cm, y, "Predicted Gas Decay Curve")
    y -= 0.4*cm
    img_x = 1.3*cm
    img_w = w - 2.6*cm
    img_h = 8.0*cm
    # Write png to temp buffer and draw
    img_buf = io.BytesIO(curve_png)
    c.drawImage(ImageReader(img_buf), img_x, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(1.3*cm, 1.0*cm, "Note: SafeMine AIR™ is a decision-support tool. Always follow site safety procedures and verify with approved gas testing before re-entry.")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# reportlab ImageReader
from reportlab.lib.utils import ImageReader


# -----------------------------
# App pages
# -----------------------------
def risk_label(t_safe_min):
    if t_safe_min <= 30:
        return "SAFE (Green)"
    if t_safe_min <= 60:
        return "CAUTION (Amber)"
    return "HAZARDOUS (Red)"

def page_home():
    st.markdown(f"<div class='safemine-hero'><h2>🛡️ {APP_NAME}</h2>"
                f"<p class='small-muted'>{SYSTEM_OVERVIEW}</p>"
                f"<span class='tag'>Hybrid ML + Physics</span>"
                f"<span class='tag'>No heavy integration</span>"
                f"<span class='tag'>Underground re-entry decision support</span>"
                f"</div>", unsafe_allow_html=True)
    st.write("")
    col1, col2 = st.columns([1.15, 1.0], gap="large")
    with col1:
        st.subheader("Problem Summary")
        st.write(PROBLEM_SUMMARY)
        st.subheader("What this demo includes")
        st.markdown(
            """
            - Simulated dataset (CO / NOx / SO₂)
            - Model training (predicts a correction factor **alpha**)
            - Hybrid decay prediction: `C(t) = C0 * exp(-(Q/V) * alpha * t)`
            - Re-entry time estimation vs a safety threshold
            - Scenario testing (fan airflow adjustments)
            - Professional PDF report generation
            """
        )
    with col2:
        st.subheader("Quick start")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.info("Tip: Use the sidebar to generate data, train the model, then run predictions.", icon="💡")


def page_data():
    st.header("📁 Data")
    st.caption("Use the bundled sample dataset or upload your own CSV to run SafeMine AIR™.")

    st.info(
        "This demo ships with a pre-generated simulated dataset in **data/synthetic_gas_decay.csv** "
        "so you can run the full workflow immediately (load → train → predict → report)."
    )

    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.subheader("Load dataset")
        use_sample = st.button("Load bundled sample dataset", type="primary")
        uploaded = st.file_uploader("Or upload a CSV dataset", type=["csv"])

        if use_sample:
            df = ensure_dataset()
            st.session_state["df"] = df
            st.success(f"Loaded sample dataset with {len(df):,} rows.")
        elif uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                # Basic validation
                missing_cols = [c for c in REQUIRED_COLS if c not in df_up.columns]
                if missing_cols:
                    st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")
                else:
                    st.session_state["df"] = df_up
                    st.success(f"Loaded uploaded dataset with {len(df_up):,} rows.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        st.markdown("---")
        st.subheader("Required columns")
        st.write(", ".join(REQUIRED_COLS))

        st.caption("Tip: You can export your mine logs to match these columns, or start with the sample dataset.")

    with colB:
        df = st.session_state.get("df", None)
        if df is None:
            df = ensure_dataset()
            st.session_state["df"] = df

        st.subheader("Quick stats")
        st.metric("Rows", f"{len(df):,}")
        if "event_id" in df.columns:
            st.metric("Events", f"{df['event_id'].nunique():,}")
        if "gas" in df.columns:
            st.metric("Gases", ", ".join(sorted(df["gas"].dropna().unique().tolist())))

        st.markdown("---")
        st.subheader("Preview")
        st.dataframe(df.head(30), use_container_width=True)

        st.markdown("---")
        st.subheader("Data quality")
        missing = df.isna().mean().sort_values(ascending=False).head(8)
        st.bar_chart(missing)

def page_train():
    st.header("🧠 Train Model")
    st.caption("Trains a model to predict the correction factor α (alpha), which adjusts the physics baseline to match observed clearance behaviour.")
    df = load_training_frame()
    st.write(f"Training frame: **{len(df):,} rows** (after removing invalid alpha estimates).")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        model_name = st.selectbox("Model type", ["RandomForest", "GradientBoosting", "MLP"], index=0)
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
        if st.button("Train & save model", type="primary"):
            with st.spinner("Training..."):
                metrics = train_and_save(model_name=model_name, test_size=float(test_size))
            st.success("Model trained and saved.")
            st.json(metrics)
    with col2:
        st.subheader("Feature columns")
        st.code(", ".join(FEATURE_COLS))
        st.subheader("What α means")
        st.markdown(
            """
            We use a physics baseline:

            **C(t) = C0 · exp(-(Q/V) · t)**

            SafeMine AIR™ learns a correction factor **α** so that:

            **C(t) = C0 · exp(-(Q/V) · α · t)**

            - **α > 1** → faster clearance than baseline  
            - **α < 1** → slower clearance than baseline  
            """
        )

    st.divider()
    st.subheader("Model status")
    model, meta = load_model()
    if model is None:
        st.warning("No trained model found yet. Train and save a model above.")
    else:
        st.success("Trained model found.")
        st.json(meta)

def page_dashboard():
    st.header("📊 Prediction Dashboard")
    model, meta = load_model()
    if model is None:
        st.warning("Train a model first (see **Train Model** in the sidebar).")
        return

    left, right = st.columns([0.95, 1.25], gap="large")
    with left:
        st.subheader("Inputs")
        gas = st.selectbox("Gas type", GASES, index=0)
        C0 = st.number_input("Initial concentration C0 (ppm)", min_value=1.0, value=250.0, step=10.0)
        threshold = st.number_input("Safety threshold (ppm)", min_value=0.1, value=50.0, step=5.0)
        Q = st.number_input("Airflow rate Q (m³/s)", min_value=1.0, value=45.0, step=1.0)
        V = st.number_input("Effective volume V (m³)", min_value=100.0, value=3500.0, step=100.0)
        duct_len = st.number_input("Duct length (m)", min_value=1.0, value=180.0, step=10.0)
        duct_diam = st.number_input("Duct diameter (m)", min_value=0.1, value=0.9, step=0.05, format="%.2f")
        aux_fans = st.slider("Number of auxiliary fans", 0, 6, 2, 1)
        temp_c = st.number_input("Temperature (°C)", min_value=-5.0, value=26.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)

        st.subheader("Scenario testing")
        airflow_mult = st.slider("Airflow multiplier (simulate fan adjustments)", 0.5, 2.0, 1.0, 0.05)
        horizon = st.slider("Prediction horizon (minutes)", 30, 240, 120, 5)

        run = st.button("Run prediction", type="primary")

    with right:
        if run:
            X = pd.DataFrame([{
                "gas": gas,
                "Q_m3s": Q,
                "V_m3": V,
                "duct_length_m": duct_len,
                "duct_diameter_m": duct_diam,
                "aux_fans": aux_fans,
                "temp_C": temp_c,
                "humidity_pct": humidity,
                "C0_ppm": C0,
            }])

            alpha = float(model.predict(X)[0])
            alpha = float(np.clip(alpha, 0.15, 2.5))

            # baseline
            t_grid = np.linspace(0, horizon, 241)
            Ct = hybrid_decay_curve(C0=C0, Q=Q, V=V, alpha=alpha, t_minutes=t_grid)
            t_safe = time_to_threshold(C0=C0, Q=Q, V=V, alpha=alpha, threshold_ppm=threshold)
            label = risk_label(t_safe)

            # scenario airflow
            Q2 = Q * float(airflow_mult)
            Ct2 = hybrid_decay_curve(C0=C0, Q=Q2, V=V, alpha=alpha, t_minutes=t_grid)
            t_safe2 = time_to_threshold(C0=C0, Q=Q2, V=V, alpha=alpha, threshold_ppm=threshold)

            # Metrics row
            m1, m2, m3 = st.columns(3, gap="medium")
            with m1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Predicted re-entry time", f"{t_safe:.1f} min")
                st.caption(f"Risk: **{label}**")
                st.markdown("</div>", unsafe_allow_html=True)
            with m2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Predicted α (alpha)", f"{alpha:.3f}")
                st.caption("Hybrid correction factor")
                st.markdown("</div>", unsafe_allow_html=True)
            with m3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric(f"Re-entry time (airflow ×{airflow_mult:.2f})", f"{t_safe2:.1f} min")
                delta = t_safe2 - t_safe
                st.caption(f"Change vs baseline: {delta:+.1f} min")
                st.markdown("</div>", unsafe_allow_html=True)

            # Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_grid, y=Ct, mode="lines", name="Predicted (baseline Q)"))
            fig.add_trace(go.Scatter(x=t_grid, y=Ct2, mode="lines", name=f"Scenario (Q×{airflow_mult:.2f})", line=dict(dash="dash")))
            fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold ({threshold} ppm)", annotation_position="top left")

            # Mark safe times
            fig.add_vline(x=t_safe, line_dash="dot", annotation_text="Re-entry (baseline)", annotation_position="top right")
            fig.add_vline(x=t_safe2, line_dash="dot", annotation_text="Re-entry (scenario)", annotation_position="bottom right")

            fig.update_layout(
                title="Predicted Gas Concentration Decay",
                xaxis_title="Time after blast (minutes)",
                yaxis_title="Concentration (ppm)",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Store in session for reports
            st.session_state["last_prediction"] = {
                "gas": gas, "C0_ppm": float(C0), "threshold_ppm": float(threshold),
                "Q_m3s": float(Q), "V_m3": float(V),
                "duct_length_m": float(duct_len), "duct_diameter_m": float(duct_diam),
                "aux_fans": int(aux_fans), "temp_C": float(temp_c), "humidity_pct": float(humidity),
                "alpha": float(alpha), "t_safe_min": float(t_safe), "risk_label": label,
                "airflow_mult": float(airflow_mult), "t_safe_scenario_min": float(t_safe2),
            }
        else:
            st.info("Fill inputs and click **Run prediction** to view results.", icon="📌")

def page_reports():
    st.header("🧾 Reports")
    st.caption("Generate a professional PDF report from the latest dashboard prediction.")
    pred = st.session_state.get("last_prediction")

    if not pred:
        st.warning("No prediction found yet. Run a prediction in **Prediction Dashboard** first.")
        return

    col1, col2 = st.columns([1.0, 1.0], gap="large")
    with col1:
        st.subheader("Report content")
        st.write("This PDF includes:")
        st.markdown(
            """
            - Summary of inputs and predicted re-entry time  
            - Safety classification (green / amber / red)  
            - Predicted gas decay curve plot  
            - Notes about decision-support use  
            """
        )
        st.json({k: pred[k] for k in ["gas","C0_ppm","threshold_ppm","Q_m3s","V_m3","alpha","t_safe_min","risk_label"]})

    with col2:
        # create a matplotlib version of the curve for embedding
        t_grid = np.linspace(0,  pred.get("horizon", 120), 241)
        # Use the last used horizon if present; else 120
        horizon = st.session_state.get("last_horizon", 120)
        t_grid = np.linspace(0, horizon, 241)
        Ct = hybrid_decay_curve(pred["C0_ppm"], pred["Q_m3s"], pred["V_m3"], pred["alpha"], t_grid)

        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ax.plot(t_grid, Ct)
        ax.axhline(pred["threshold_ppm"], linestyle="--")
        ax.axvline(pred["t_safe_min"], linestyle=":")
        ax.set_xlabel("Time after blast (minutes)")
        ax.set_ylabel("Concentration (ppm)")
        ax.set_title("Predicted Gas Decay Curve")
        curve_png = _fig_to_png_bytes(fig)

        if st.button("Generate PDF report", type="primary"):
            # update with horizon in payload
            payload = dict(pred)
            payload["horizon"] = float(horizon)
            pdf_bytes = generate_pdf_report(payload, curve_png)
            st.success("Report generated.")
            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name=f"safemine_air_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

        st.subheader("Preview (plot used in report)")
        st.image(curve_png, use_column_width=True)


def page_about():
    st.header("ℹ️ About SafeMine AIR™")
    st.write(
        "SafeMine AIR™ is a predictive decision-support tool that estimates safe re-entry times in underground mines "
        "after blasting by modelling toxic gas decay and ventilation performance."
    )

    st.subheader("Why it matters")
    st.markdown(
        """
- Time-based re-entry rules can be overly conservative or unsafe when conditions change.
- Manual gas checks are often irregular and reactive.
- Ventilation logs and gas readings are underutilised for prediction and planning.

**SafeMine AIR™** brings a hybrid approach (machine learning + ventilation physics) to provide fast, explainable, data-driven re-entry guidance.
        """
    )

    st.markdown("---")
    st.subheader("Built by Navon Labs")

    cols = st.columns([0.35, 0.65], gap="large")
    with cols[0]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_column_width=True)
    with cols[1]:
        st.write(
            "Navon Labs (Pty) Ltd is a Botswana-based AI and software consultancy that helps organisations transform raw data "
            "into actionable insights by combining expertise in AI, data engineering, and software development."
        )
        st.write("**Vision:** To be Africa’s leading AI-driven innovation lab, enabling sustainable growth and operational excellence.")
        st.markdown(
            """
**Core services:**
- AI strategy and ROI-focused use-case identification
- Data engineering and analytics foundations
- Machine learning development and deployment
            """
        )

    st.markdown("---")
    st.subheader("Contact")
    st.markdown(
        """
- **Location:** Palapye, Khurumela (Botswana)
- **Email:** consult@navonlabs.co.bw; ounas.saubi@gmail.com
- **Website:** motsadi.github.io/navonlabs-site/
        """
    )



# Sidebar navigation
# -----------------------------
st.sidebar.title(APP_NAME)
page = render_branding()

st.sidebar.radio(
    "Navigate",
    ["Home", "Data", "Train Model", "Prediction Dashboard", "Reports", "About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Demo workflow")
st.sidebar.markdown(
    """
    1) Load data → **Data**  
    2) Train model → **Train Model**  
    3) Run prediction → **Prediction Dashboard**  
    4) Export report → **Reports**
    """
)

# Run page
if page == "Home":
    page_home()
elif page == "Data":
    page_data()
elif page == "Train Model":
    page_train()
elif page == "Prediction Dashboard":
    page_dashboard()
elif page == "Reports":
    page_reports()
else:
    page_about()
