
import io
import os
import math
import json
import time
import base64
import textwrap
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
# Data schema (required columns)
# -----------------------------
REQUIRED_COLS = [
    "event_id",
    "timestamp",
    "gas",
    "Q_m3s",
    "V_m3",
    "duct_length_m",
    "duct_diameter_m",
    "aux_fans",
    "temp_C",
    "humidity_pct",
    "C0_ppm",
    "t_min",
    "Ct_ppm",
]

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
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
          radial-gradient(circle at top right, rgba(30, 64, 175, 0.14), transparent 24%),
          linear-gradient(180deg, #f5f7fb 0%, #eef4f7 100%);
      }
      .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
      .safemine-hero {
        padding: 1.25rem 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(8, 47, 73, 0.96), rgba(15, 118, 110, 0.92));
        color: #f8fafc;
        border: 1px solid rgba(15, 23, 42, 0.12);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
      }
      .metric-card {
        padding: 1rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: rgba(255,255,255,0.82);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
      }
      .section-card {
        padding: 1.05rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: rgba(255,255,255,0.74);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
      }
      .status-banner {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 0.8rem;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.10);
      }
      .status-safe { background: linear-gradient(135deg, #047857, #10b981); }
      .status-marginal { background: linear-gradient(135deg, #b45309, #f59e0b); }
      .status-insufficient { background: linear-gradient(135deg, #b91c1c, #ef4444); }
      .small-muted { color: rgba(15, 23, 42, 0.64); font-size: 0.94rem; }
      .hero-muted { color: rgba(241, 245, 249, 0.90); font-size: 0.97rem; }
      .tag {
        display:inline-block; padding: 0.24rem 0.65rem; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.20); background: rgba(255,255,255,0.12);
        color: #f8fafc; font-size: 0.82rem; margin-right: 0.38rem; margin-top: 0.35rem;
      }
      .mini-stat {
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        min-height: 92px;
      }
      .mini-stat .label {
        color: rgba(226, 232, 240, 0.84);
        font-size: 0.83rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .mini-stat .value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.35rem;
      }
      .mini-stat .detail {
        color: rgba(226, 232, 240, 0.88);
        font-size: 0.84rem;
        margin-top: 0.2rem;
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


TARGET_CLEARANCE_MIN = 45.0


def risk_label(t_safe_min):
    if t_safe_min <= 30:
        return "SAFE (Green)"
    if t_safe_min <= 60:
        return "CAUTION (Amber)"
    return "HAZARDOUS (Red)"


def risk_theme(label):
    if label.startswith("SAFE"):
        return {
            "status": "status-safe",
            "accent": "#10b981",
            "title": "Ventilation plan is within target range.",
        }
    if label.startswith("CAUTION"):
        return {
            "status": "status-marginal",
            "accent": "#f59e0b",
            "title": "Conditions are workable but need close control.",
        }
    return {
        "status": "status-insufficient",
        "accent": "#ef4444",
        "title": "Ventilation is not sufficient for the desired re-entry window.",
    }


def estimate_duct_efficiency(duct_length_m, duct_diameter_m):
    length_term = np.clip(np.exp(-(max(float(duct_length_m), 1.0) - 120.0) / 500.0), 0.60, 1.15)
    diameter_term = np.clip((max(float(duct_diameter_m), 0.1) / 0.90) ** 0.40, 0.70, 1.35)
    return float(np.clip(length_term * diameter_term, 0.55, 1.20))


def required_airflow_for_target(C0, V, alpha, threshold_ppm, target_min=TARGET_CLEARANCE_MIN):
    threshold_ppm = max(float(threshold_ppm), 1e-6)
    alpha = max(float(alpha), 1e-6)
    if threshold_ppm >= C0:
        return 0.0
    t_sec = max(float(target_min), 1.0) * 60.0
    q_needed = - (float(V) / (alpha * t_sec)) * math.log(threshold_ppm / float(C0))
    return max(float(q_needed), 0.0)


def assess_ventilation(inputs, alpha, t_safe_min, threshold_ppm, target_clearance_min=TARGET_CLEARANCE_MIN):
    Q = float(inputs["Q_m3s"])
    V = float(inputs["V_m3"])
    duct_len = float(inputs["duct_length_m"])
    duct_diam = float(inputs["duct_diameter_m"])
    aux_fans = int(inputs["aux_fans"])
    temp_c = float(inputs["temp_C"])
    humidity = float(inputs["humidity_pct"])

    duct_efficiency = estimate_duct_efficiency(duct_len, duct_diam)
    fan_assist_factor = 1.0 + 0.08 * aux_fans
    effective_airflow = Q * duct_efficiency * fan_assist_factor
    air_changes_per_hour = effective_airflow * 3600.0 / max(V, 1.0)
    required_airflow = required_airflow_for_target(
        C0=inputs["C0_ppm"],
        V=V,
        alpha=alpha,
        threshold_ppm=threshold_ppm,
        target_min=target_clearance_min,
    )
    airflow_gap = required_airflow - effective_airflow
    airflow_ratio = effective_airflow / max(required_airflow, 1e-6) if required_airflow > 0 else 1.0
    target_gap_min = float(t_safe_min) - float(target_clearance_min)

    if airflow_gap > 6.0 or target_gap_min > 15 or air_changes_per_hour < 25:
        status = "INSUFFICIENT"
        message = "Re-entry will likely be delayed unless ventilation delivery is improved."
    elif airflow_gap > 1.0 or target_gap_min > 0 or air_changes_per_hour < 35:
        status = "MARGINAL"
        message = "Ventilation is close to acceptable, but the margin for safe clearance is narrow."
    else:
        status = "ADEQUATE"
        message = "Ventilation delivery is aligned with the target re-entry window."

    recommendations = []
    if airflow_gap > 0.5:
        recommendations.append(
            f"Increase delivered airflow by about {max(airflow_gap, 0.0):.1f} m3/s to achieve a {target_clearance_min:.0f}-minute target."
        )
    if duct_len > 220:
        recommendations.append("Shorten the active duct run or reduce leakage to improve delivery at the face.")
    if duct_diam < 0.85:
        recommendations.append("Consider a larger duct diameter to reduce resistance and increase usable airflow.")
    if aux_fans < 4 and status != "ADEQUATE":
        recommendations.append("Add or reposition an auxiliary fan closer to the face to improve mixing and clearance.")
    if temp_c > 32 or humidity > 78:
        recommendations.append("Increase gas-check frequency because hot or humid conditions can worsen operator exposure risk.")
    recommendations.append("Keep re-entry locked out until field gas tests confirm concentrations are below the site threshold.")

    return {
        "status": status,
        "message": message,
        "duct_efficiency": float(duct_efficiency),
        "fan_assist_factor": float(fan_assist_factor),
        "effective_airflow_m3s": float(effective_airflow),
        "air_changes_per_hour": float(air_changes_per_hour),
        "required_airflow_m3s": float(required_airflow),
        "airflow_gap_m3s": float(airflow_gap),
        "airflow_ratio": float(airflow_ratio),
        "target_gap_min": float(target_gap_min),
        "recommendations": recommendations[:5],
    }


def predict_alpha(model, features):
    X = pd.DataFrame([features])
    alpha = float(model.predict(X)[0])
    return float(np.clip(alpha, 0.15, 2.5))


def build_scenario_table(model, base_inputs, threshold_ppm, baseline_t_safe):
    scenario_specs = [
        ("Current design", {}, "Current operating point"),
        ("Increase airflow by 25%", {"Q_m3s": base_inputs["Q_m3s"] * 1.25}, "Boost primary fan delivery"),
        (
            "Add one auxiliary fan",
            {"aux_fans": min(base_inputs["aux_fans"] + 1, 6), "Q_m3s": base_inputs["Q_m3s"] * 1.08},
            "Improves local dilution and delivery",
        ),
        (
            "Increase duct diameter by 0.15 m",
            {"duct_diameter_m": min(base_inputs["duct_diameter_m"] + 0.15, 2.0), "Q_m3s": base_inputs["Q_m3s"] * 1.10},
            "Reduces duct resistance",
        ),
        (
            "Shorten duct by 50 m",
            {"duct_length_m": max(base_inputs["duct_length_m"] - 50.0, 5.0), "Q_m3s": base_inputs["Q_m3s"] * 1.05},
            "Cuts delivery losses",
        ),
    ]

    rows = []
    for name, overrides, note in scenario_specs:
        scenario_inputs = dict(base_inputs)
        scenario_inputs.update(overrides)
        alpha = predict_alpha(model, scenario_inputs)
        t_safe = time_to_threshold(
            C0=scenario_inputs["C0_ppm"],
            Q=scenario_inputs["Q_m3s"],
            V=scenario_inputs["V_m3"],
            alpha=alpha,
            threshold_ppm=threshold_ppm,
        )
        ventilation = assess_ventilation(scenario_inputs, alpha, t_safe, threshold_ppm)
        rows.append(
            {
                "Scenario": name,
                "Predicted re-entry (min)": round(t_safe, 1),
                "Time saved (min)": round(float(baseline_t_safe) - float(t_safe), 1),
                "Effective airflow (m3/s)": round(ventilation["effective_airflow_m3s"], 1),
                "Ventilation status": ventilation["status"].title(),
                "Note": note,
            }
        )

    df = pd.DataFrame(rows)
    df["rank"] = df["Predicted re-entry (min)"].rank(method="dense")
    return df.drop(columns=["rank"])


def build_clearance_gauge(t_safe_min):
    gauge_max = max(120, math.ceil(float(t_safe_min) / 10.0) * 10)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=float(t_safe_min),
            number={"suffix": " min"},
            delta={"reference": TARGET_CLEARANCE_MIN, "relative": False},
            title={"text": "Predicted re-entry time"},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, gauge_max], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#1d4ed8", "width": 3}, "thickness": 0.8, "value": TARGET_CLEARANCE_MIN},
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_airflow_gap_chart(ventilation):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Delivered airflow", "Required airflow"],
            y=[ventilation["effective_airflow_m3s"], ventilation["required_airflow_m3s"]],
            marker_color=["#0f766e", "#1d4ed8"],
            text=[f"{ventilation['effective_airflow_m3s']:.1f}", f"{ventilation['required_airflow_m3s']:.1f}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Airflow sufficiency check",
        yaxis_title="m3/s",
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
    )
    return fig


def _pdf_draw_lines(c, x, y, lines, font="Helvetica", size=9.2, color=colors.black, leading=0.45 * cm):
    c.setFillColor(color)
    c.setFont(font, size)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


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

    c.setFillColor(colors.HexColor("#0B3D91"))
    c.rect(0, h - 2.2 * cm, w, 2.2 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.3 * cm, h - 1.35 * cm, "SafeMine AIR™ – Re-entry Prediction Report")
    c.setFont("Helvetica", 9)
    c.drawString(1.3 * cm, h - 1.85 * cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    ventilation = payload["ventilation"]
    scenarios = payload["scenario_rows"]

    y = h - 3.0 * cm
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3 * cm, y, "Executive summary")
    y -= 0.65 * cm

    summary_lines = [
        f"Gas: {payload['gas']} | C0: {payload['C0_ppm']:.1f} ppm | Threshold: {payload['threshold_ppm']:.1f} ppm",
        f"Predicted re-entry time: {payload['t_safe_min']:.1f} minutes | Risk: {payload['risk_label']}",
        f"Ventilation status: {ventilation['status']} | Effective airflow: {ventilation['effective_airflow_m3s']:.1f} m3/s",
        f"Required airflow for {TARGET_CLEARANCE_MIN:.0f}-minute target: {ventilation['required_airflow_m3s']:.1f} m3/s",
        f"Scenario airflow multiplier tested: x{payload['airflow_mult']:.2f} | Scenario re-entry: {payload['t_safe_scenario_min']:.1f} minutes",
    ]
    wrapped_lines = []
    for line in summary_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=96))
    y = _pdf_draw_lines(c, 1.3 * cm, y, wrapped_lines, size=9.3)

    y -= 0.1 * cm
    c.setStrokeColor(colors.HexColor("#D0D7DE"))
    c.line(1.3 * cm, y, w - 1.3 * cm, y)
    y -= 0.65 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3 * cm, y, "Inputs")
    y -= 0.6 * cm

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
    tbl = Table(table_data, colWidths=[6.0 * cm, 9.0 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F6F8FA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBFCFD")]),
            ]
        )
    )
    tbl.wrapOn(c, w, h)
    tbl_h = 0.45 * cm * len(table_data)
    tbl.drawOn(c, 1.3 * cm, y - tbl_h + 0.2 * cm)
    y -= tbl_h + 0.7 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3 * cm, y, "Ventilation diagnostics")
    y -= 0.55 * cm
    diagnostics = [
        f"Status: {ventilation['status']}",
        f"Message: {ventilation['message']}",
        f"Effective airflow: {ventilation['effective_airflow_m3s']:.1f} m3/s",
        f"Required airflow for target: {ventilation['required_airflow_m3s']:.1f} m3/s",
        f"Airflow gap: {ventilation['airflow_gap_m3s']:.1f} m3/s",
        f"Air changes per hour: {ventilation['air_changes_per_hour']:.1f}",
    ]
    wrapped_diag = []
    for line in diagnostics:
        wrapped_diag.extend(textwrap.wrap(line, width=96))
    y = _pdf_draw_lines(c, 1.3 * cm, y, wrapped_diag)

    y -= 0.1 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3 * cm, y, "Recommended actions")
    y -= 0.55 * cm
    reco_lines = []
    for rec in ventilation["recommendations"][:4]:
        reco_lines.extend(textwrap.wrap(f"- {rec}", width=96))
    y = _pdf_draw_lines(c, 1.3 * cm, y, reco_lines)

    y -= 0.1 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.3 * cm, y, "Improvement scenarios")
    y -= 0.55 * cm
    scenario_data = [["Scenario", "Re-entry", "Time saved"]]
    for row in scenarios[:4]:
        scenario_data.append(
            [
                row["Scenario"],
                f"{row['Predicted re-entry (min)']:.1f} min",
                f"{row['Time saved (min)']:+.1f} min",
            ]
        )
    scenario_tbl = Table(scenario_data, colWidths=[8.2 * cm, 3.2 * cm, 3.4 * cm])
    scenario_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#93C5FD")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    scenario_tbl.wrapOn(c, w, h)
    scenario_tbl.drawOn(c, 1.3 * cm, max(y - 2.7 * cm, 3.0 * cm))

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(
        1.3 * cm,
        1.0 * cm,
        "Note: SafeMine AIR™ is decision support only. Follow approved site procedures and verify with gas testing before re-entry.",
    )
    c.showPage()

    c.setFillColor(colors.HexColor("#0B3D91"))
    c.rect(0, h - 2.0 * cm, w, 2.0 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(1.3 * cm, h - 1.25 * cm, "Predicted Gas Decay Curve")
    c.setFont("Helvetica", 9)
    c.drawString(1.3 * cm, h - 1.7 * cm, "Baseline and what-if airflow scenario")

    img_x = 1.3 * cm
    img_y = 4.0 * cm
    img_w = w - 2.6 * cm
    img_h = h - 7.0 * cm
    img_buf = io.BytesIO(curve_png)
    c.drawImage(ImageReader(img_buf), img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(1.3 * cm, 1.0 * cm, "Use this chart with site ventilation standards, gas-testing logs, and shift control procedures.")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# reportlab ImageReader
from reportlab.lib.utils import ImageReader


def page_home():
    df = ensure_dataset()
    model, meta = load_model()
    model_status = meta["model_name"] if meta else "Not yet trained"
    st.markdown(
        f"""
        <div class='safemine-hero'>
          <h2 style='margin-bottom:0.35rem;'>🛡️ {APP_NAME}</h2>
          <p class='hero-muted'>{SYSTEM_OVERVIEW}</p>
          <div style='display:flex; gap:0.9rem; margin:1rem 0 0.6rem 0; flex-wrap:wrap;'>
            <div class='mini-stat'>
              <div class='label'>Sample events</div>
              <div class='value'>{df['event_id'].nunique():,}</div>
              <div class='detail'>Synthetic blast records available to train and test predictions.</div>
            </div>
            <div class='mini-stat'>
              <div class='label'>Model status</div>
              <div class='value'>{model_status}</div>
              <div class='detail'>Train once, then use the dashboard to compare baseline vs improved ventilation plans.</div>
            </div>
            <div class='mini-stat'>
              <div class='label'>Target clearance</div>
              <div class='value'>{TARGET_CLEARANCE_MIN:.0f} min</div>
              <div class='detail'>The app flags insufficient ventilation when the predicted plan misses this operational target.</div>
            </div>
          </div>
          <span class='tag'>Hybrid ML + Physics</span>
          <span class='tag'>Ventilation alerts</span>
          <span class='tag'>Scenario-based reporting</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns([1.15, 1.0], gap="large")
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Why this matters")
        st.write(PROBLEM_SUMMARY)
        st.markdown(
            """
            - Replace fixed waiting periods with a forecast tied to gas concentration, airflow, and ventilation geometry.
            - Detect when the current layout is unlikely to clear fumes in the target window.
            - Turn every prediction into an operational report with recommended ventilation actions.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("What is new in this version")
        st.markdown(
            """
            - A more visual prediction dashboard with clearer status cards and gauges.
            - Ventilation sufficiency diagnostics showing delivered airflow, required airflow, and the gap.
            - Improvement scenarios to show likely time savings before crews re-enter.
            - Richer reports with actions, scenario comparisons, and executive-ready summaries.
            """
        )
        st.subheader("Quick start")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.info("Tip: Use the sidebar to load data, train the model, then run predictions.", icon="💡")
        st.markdown("</div>", unsafe_allow_html=True)


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
            base_inputs = {
                "gas": gas,
                "Q_m3s": Q,
                "V_m3": V,
                "duct_length_m": duct_len,
                "duct_diameter_m": duct_diam,
                "aux_fans": aux_fans,
                "temp_C": temp_c,
                "humidity_pct": humidity,
                "C0_ppm": C0,
            }

            alpha = predict_alpha(model, base_inputs)

            t_grid = np.linspace(0, horizon, 241)
            Ct = hybrid_decay_curve(C0=C0, Q=Q, V=V, alpha=alpha, t_minutes=t_grid)
            t_safe = time_to_threshold(C0=C0, Q=Q, V=V, alpha=alpha, threshold_ppm=threshold)
            label = risk_label(t_safe)
            theme = risk_theme(label)
            ventilation = assess_ventilation(base_inputs, alpha, t_safe, threshold)

            Q2 = Q * float(airflow_mult)
            Ct2 = hybrid_decay_curve(C0=C0, Q=Q2, V=V, alpha=alpha, t_minutes=t_grid)
            t_safe2 = time_to_threshold(C0=C0, Q=Q2, V=V, alpha=alpha, threshold_ppm=threshold)
            scenarios = build_scenario_table(model, base_inputs, threshold, t_safe)
            best_row = scenarios.sort_values("Predicted re-entry (min)").iloc[0].to_dict()

            st.markdown(
                f"""
                <div class="status-banner {theme['status']}">
                  <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.25rem;">{theme['title']}</div>
                  <div style="font-size:0.94rem;">
                    {ventilation['message']} Current predicted re-entry is <strong>{t_safe:.1f} minutes</strong>,
                    compared with the operational target of <strong>{TARGET_CLEARANCE_MIN:.0f} minutes</strong>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4, gap="medium")
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
                st.metric("Effective airflow", f"{ventilation['effective_airflow_m3s']:.1f} m3/s")
                st.caption(f"Gap vs target: {ventilation['airflow_gap_m3s']:+.1f} m3/s")
                st.markdown("</div>", unsafe_allow_html=True)
            with m4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric(f"Re-entry time (airflow x{airflow_mult:.2f})", f"{t_safe2:.1f} min")
                delta = t_safe2 - t_safe
                st.caption(f"Change vs baseline: {delta:+.1f} min")
                st.markdown("</div>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_grid, y=Ct, mode="lines", name="Predicted (baseline Q)", line=dict(color="#0f766e", width=4)))
            fig.add_trace(
                go.Scatter(
                    x=t_grid,
                    y=Ct2,
                    mode="lines",
                    name=f"Scenario (Qx{airflow_mult:.2f})",
                    line=dict(color="#1d4ed8", width=3, dash="dash"),
                )
            )
            fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold ({threshold} ppm)", annotation_position="top left")
            fig.add_vline(x=t_safe, line_dash="dot", annotation_text="Re-entry (baseline)", annotation_position="top right")
            fig.add_vline(x=t_safe2, line_dash="dot", annotation_text="Re-entry (scenario)", annotation_position="bottom right")
            fig.update_layout(
                title="Predicted Gas Concentration Decay",
                xaxis_title="Time after blast (minutes)",
                yaxis_title="Concentration (ppm)",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.86)",
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns([0.95, 1.05], gap="large")
            with c1:
                st.plotly_chart(build_clearance_gauge(t_safe), use_container_width=True)
            with c2:
                st.plotly_chart(build_airflow_gap_chart(ventilation), use_container_width=True)

            st.subheader("Ventilation diagnosis")
            d1, d2 = st.columns([1.1, 0.9], gap="large")
            with d1:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    - **Status:** {ventilation['status']}
                    - **Air changes per hour:** {ventilation['air_changes_per_hour']:.1f}
                    - **Duct efficiency factor:** {ventilation['duct_efficiency']:.2f}
                    - **Required airflow for target:** {ventilation['required_airflow_m3s']:.1f} m3/s
                    - **Best tested scenario:** {best_row['Scenario']} ({best_row['Predicted re-entry (min)']:.1f} min)
                    """
                )
                st.markdown("**Recommended actions**")
                for rec in ventilation["recommendations"]:
                    st.write(f"- {rec}")
                st.markdown("</div>", unsafe_allow_html=True)
            with d2:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Improvement scenarios")
                st.dataframe(
                    scenarios[["Scenario", "Predicted re-entry (min)", "Time saved (min)", "Ventilation status"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.session_state["last_prediction"] = {
                "gas": gas, "C0_ppm": float(C0), "threshold_ppm": float(threshold),
                "Q_m3s": float(Q), "V_m3": float(V),
                "duct_length_m": float(duct_len), "duct_diameter_m": float(duct_diam),
                "aux_fans": int(aux_fans), "temp_C": float(temp_c), "humidity_pct": float(humidity),
                "alpha": float(alpha), "t_safe_min": float(t_safe), "risk_label": label,
                "airflow_mult": float(airflow_mult), "t_safe_scenario_min": float(t_safe2),
                "ventilation": ventilation,
                "scenario_rows": scenarios.to_dict(orient="records"),
                "horizon": float(horizon),
            }
            st.session_state["last_horizon"] = float(horizon)
        else:
            st.info("Fill inputs and click **Run prediction** to view results.", icon="📌")

def page_reports():
    st.header("🧾 Reports")
    st.caption("Generate a richer operations report from the latest dashboard prediction.")
    pred = st.session_state.get("last_prediction")

    if not pred:
        st.warning("No prediction found yet. Run a prediction in **Prediction Dashboard** first.")
        return

    ventilation = pred["ventilation"]
    scenarios = pd.DataFrame(pred["scenario_rows"])
    theme = risk_theme(pred["risk_label"])

    st.markdown(
        f"""
        <div class="status-banner {theme['status']}">
          <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.25rem;">Reporting summary</div>
          <div style="font-size:0.94rem;">
            Latest prediction: <strong>{pred['t_safe_min']:.1f} minutes</strong> for {pred['gas']},
            with ventilation status <strong>{ventilation['status']}</strong>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.95, 1.05], gap="large")
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Executive summary")
        st.markdown(
            """
            - Executive summary of risk, re-entry forecast, and ventilation adequacy.
            - Airflow sufficiency gap against the target clearance window.
            - Improvement scenarios to show which changes save the most time.
            - Action recommendations ready for supervisors or shift reports.
            """
        )
        st.json(
            {
                "gas": pred["gas"],
                "predicted_re_entry_min": round(pred["t_safe_min"], 1),
                "risk_label": pred["risk_label"],
                "ventilation_status": ventilation["status"],
                "effective_airflow_m3s": round(ventilation["effective_airflow_m3s"], 1),
                "required_airflow_m3s": round(ventilation["required_airflow_m3s"], 1),
            }
        )
        st.markdown("**Recommended actions**")
        for rec in ventilation["recommendations"]:
            st.write(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        horizon = pred.get("horizon", st.session_state.get("last_horizon", 120))
        t_grid = np.linspace(0, horizon, 241)
        Ct = hybrid_decay_curve(pred["C0_ppm"], pred["Q_m3s"], pred["V_m3"], pred["alpha"], t_grid)
        Q2 = pred["Q_m3s"] * pred["airflow_mult"]
        Ct2 = hybrid_decay_curve(pred["C0_ppm"], Q2, pred["V_m3"], pred["alpha"], t_grid)

        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ax.plot(t_grid, Ct, linewidth=2.6, color="#0f766e", label="Baseline")
        ax.plot(t_grid, Ct2, linewidth=2.0, linestyle="--", color="#1d4ed8", label=f"Scenario x{pred['airflow_mult']:.2f}")
        ax.axhline(pred["threshold_ppm"], linestyle="--", color="#475569")
        ax.axvline(pred["t_safe_min"], linestyle=":", color="#ef4444")
        ax.set_xlabel("Time after blast (minutes)")
        ax.set_ylabel("Concentration (ppm)")
        ax.set_title("Predicted Gas Decay Curve")
        ax.legend()
        curve_png = _fig_to_png_bytes(fig)

        if st.button("Generate PDF report", type="primary"):
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

        json_bytes = json.dumps(pred, indent=2).encode("utf-8")
        st.download_button(
            label="Download JSON summary",
            data=json_bytes,
            file_name=f"safemine_air_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )

        st.subheader("Preview (plot used in report)")
        st.image(curve_png, use_column_width=True)

    st.subheader("Scenario comparison")
    st.dataframe(
        scenarios[["Scenario", "Predicted re-entry (min)", "Time saved (min)", "Effective airflow (m3/s)", "Ventilation status", "Note"]],
        use_container_width=True,
        hide_index=True,
    )


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
            "into actionable insights by combining expertise in AI, data engineering, and software development"
            "with a strong focus on mining and engineering applications."
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
- **Email:** consult@navonlab.com; ounas.saubi@gmail.com
- **Website:** www.navonlab.com
        """
    )



# Sidebar navigation
# -----------------------------
st.sidebar.title(APP_NAME)
render_branding()

PAGES = ["Home", "Data", "Train Model", "Prediction Dashboard", "Reports", "About"]
page = st.sidebar.radio("Navigate", PAGES, key="nav_page")


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
