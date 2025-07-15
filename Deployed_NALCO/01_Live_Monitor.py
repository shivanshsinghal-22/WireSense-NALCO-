import os
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ───────────────────────── optional auto-refresh helper ───────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH_OK = True
except ModuleNotFoundError:
    _AUTOREFRESH_OK = False
# ──────────────────────────────────────────────────────────────────────────
# PARAMETERS YOU MIGHT TWEAK
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR = ""                 # where the .pkl models live
MODEL_FILES = {
    "   UTS":          "xgboost_model_output_   UTS.pkl",
    "Conductivity": "xgboost_model_output_Conductivity.pkl",
    "Elongation":   "xgboost_model_output_Elongation.pkl",
}

EXPECTED_FEATURES: List[str] = [
    # 16 process columns
    "EMUL_OIL_L_TEMP_PV_VAL0", "STAND_OIL_L_TEMP_PV_REAL_VAL0",
    "GEAR_OIL_L_TEMP_PV_REAL_VAL0", "EMUL_OIL_L_PR_VAL0",
    "QUENCH_CW_FLOW_EXIT_VAL0", "CAST_WHEEL_RPM_VAL0",
    "BAR_TEMP_VAL0", "QUENCH_CW_FLOW_ENTRY_VAL0", "GEAR_OIL_L_PR_VAL0",
    "STANDS_OIL_L_PR_VAL0", "TUNDISH_TEMP_VAL0", "RM_MOTOR_COOL_WATER__VAL0",
    "ROLL_MILL_AMPS_VAL0", "RM_COOL_WATER_FLOW_VAL0",
    "EMULSION_LEVEL_ANALO_VAL0",
    # chemistry + furnace
    "%SI", "%FE", "%TI", "%V", "%AL", "Furnace_Temperature",
]

TARGET_COLUMNS = list(MODEL_FILES.keys())   # used only to de-normalise preds

FILL_VALUE          = 1.90
DEFAULT_REFRESH_SEC = 3        # how fast the tape moves
WINDOW_ROWS         = 5        # always show exactly 5 rows
MAX_POINTS          = 500      # not critical with small windows

# ───────────────────────── STREAMLIT PAGE CONFIG ─────────────────────────
st.set_page_config(page_title="21 → 3 Live Monitor", layout="wide")
st.title("Parameters Live Monitor")

# ───────────────────────── SIDEBAR INPUTS ────────────────────────────────
st.sidebar.header("Configuration")

# csv_path = st.sidebar.text_input(
#     "Path to *live* CSV (sliding window source)",
#     value="Real_final_Data_With_anomaly.csv",
# )

csv_path = "Real_final_Data_With_anomaly.csv"

# scaler_csv_path = st.sidebar.text_input(
#     "Path to *training* CSV (for min–max scaling)",
#     value="Training_Dataset.csv",
#     help="The file used when the models were trained. It provides\n"
#          "feature-wise and target-wise min/max values.",
# )

scaler_csv_path = "Training_Dataset.csv"

refresh_rate = st.sidebar.number_input(
    "Refresh every N seconds",
    min_value=0, max_value=60, value=DEFAULT_REFRESH_SEC, step=1,
    help="0 disables auto-advance",
)

# with st.sidebar.expander("Model directory & files"):
#     st.write(f"Model directory: `{MODEL_DIR or os.getcwd()}`")
#     for k, v in MODEL_FILES.items():
#         st.write(f"• **{k}** → `{v}`")

if refresh_rate > 0 and _AUTOREFRESH_OK:
    st_autorefresh(interval=refresh_rate * 1000, key="data_refresh")

# ───────────────────────── HELPERS ───────────────────────────────────────
def load_models(model_dir: str, files: Dict[str, str]) -> Dict[str, joblib]:
    if "models" not in st.session_state:
        models = {}
        for name, fname in files.items():
            path = os.path.join(model_dir, fname)
            if not os.path.isfile(path):
                st.error(f"❌ Model file not found: `{path}`")
                st.stop()
            models[name] = joblib.load(path)
        st.session_state.models = models
    return st.session_state.models

def read_csv_once(path: str, key: str) -> pd.DataFrame:
    """Read any CSV exactly once per session, cached by `key`."""
    if key not in st.session_state:
        st.session_state[key] = pd.read_csv(path)
    return st.session_state[key]

def clean_numeric(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
         .str.replace(r"[%,]", "", regex=True)
         .str.replace(r"[^\d\.\-eE+]", "", regex=True)
         .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")

def predict(df_feat: pd.DataFrame, models: Dict[str, joblib]) -> pd.DataFrame:
    preds = {}
    for name, model in models.items():
        try:
            order = model.get_booster().feature_names
        except AttributeError:
            order = getattr(model, "feature_names_in_", list(df_feat.columns))
        preds[name] = model.predict(df_feat[order])
    return pd.DataFrame(preds, index=df_feat.index)

def compute_minmax(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.Series]:
    """Cache & return Series of min and max for given cols."""
    key = ("minmax", id(df), tuple(cols))
    if key not in st.session_state:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"❌ Columns {missing} missing in training CSV.")
            st.stop()
        st.session_state[key] = {
            "min": df[cols].min(),
            "max": df[cols].max(),
        }
    return st.session_state[key]

def minmax_scale(df: pd.DataFrame, s_min: pd.Series, s_max: pd.Series) -> pd.DataFrame:
    scale = (s_max - s_min).replace({0: np.nan})
    df_scaled = (df - s_min) / scale
    return df_scaled.fillna(0.0)

def minmax_inverse(df_norm: pd.DataFrame, s_min: pd.Series, s_max: pd.Series) -> pd.DataFrame:
    return df_norm * (s_max - s_min) + s_min

# ───────────────────────── CHECK FILES EXIST ─────────────────────────────
if csv_path == "":
    st.info("Enter the *live* CSV path in the sidebar to begin.")
    st.stop()
if not os.path.exists(csv_path):
    st.error(f"❌ Live CSV file not found at `{csv_path}`")
    st.stop()

if scaler_csv_path == "":
    st.info("Enter the *training* CSV path in the sidebar to begin.")
    st.stop()
if not os.path.exists(scaler_csv_path):
    st.error(f"❌ Training CSV file not found at `{scaler_csv_path}`")
    st.stop()

# ───────────────────────── LOAD DATA & MODELS ────────────────────────────
live_df    = read_csv_once(csv_path,        "live_df")
training_df = read_csv_once(scaler_csv_path, "training_df")
num_rows   = len(live_df)
models     = load_models(MODEL_DIR, MODEL_FILES)

# pre-compute min/max from training CSV
feat_scaler  = compute_minmax(training_df, EXPECTED_FEATURES)
targ_scaler  = compute_minmax(training_df, TARGET_COLUMNS)

# ───────────────────────── SLIDING-WINDOW LOGIC ──────────────────────────
if "win_ptr" not in st.session_state:
    st.session_state.win_ptr = 0

start = st.session_state.win_ptr
end   = start + WINDOW_ROWS
if end > num_rows:                           # reached the end → loop
    start, end = 0, WINDOW_ROWS
    st.session_state.win_ptr = 0

window_df = live_df.iloc[start:end].reset_index(drop=True)
st.session_state.win_ptr += 1                # advance for NEXT refresh

# ───────────────────────── FEATURE PREPARATION ───────────────────────────
feature_df = window_df.copy()
for col in EXPECTED_FEATURES:
    if col not in feature_df.columns:
        feature_df[col] = FILL_VALUE
feature_df = feature_df[EXPECTED_FEATURES]

# clean → numeric
feature_df = feature_df.apply(clean_numeric, axis=0)

# NORMALISE with *training* min–max
feature_df_norm = minmax_scale(feature_df, feat_scaler["min"], feat_scaler["max"])

# ───────────────────────── PREDICTION ────────────────────────────────────
pred_df_norm = predict(feature_df_norm, models)

# DE-NORMALISE predictions for interpretability
pred_df = minmax_inverse(pred_df_norm, targ_scaler["min"], targ_scaler["max"])

# ───────────────────────── VISUALISATION ─────────────────────────────────
with st.expander(f" Rows {start+1} – {end} of {num_rows}", expanded=True):
    grid_cols = st.columns(3)
    for i, col_name in enumerate(EXPECTED_FEATURES):
        y = feature_df[col_name]              # display original scale
        with grid_cols[i % 3]:
            if y.count() < 2:
                st.caption(f"*{col_name}: no numeric data*")
            else:
                fig = go.Figure(
                    go.Scatter(
                        x=y.index, y=y.values,
                        mode="lines+markers",
                        name=col_name,
                    )
                )
                fig.update_layout(
                    title=col_name,
                    height=180,
                    margin=dict(l=8, r=8, t=30, b=20),
                    showlegend=False,
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        if i % 3 == 2 and i != len(EXPECTED_FEATURES) - 1:
            grid_cols = st.columns(3)

st.subheader("Predictions on Current Window")
pred_cols = st.columns(3)
for i, target in enumerate(pred_df.columns):
    y = pred_df[target]
    with pred_cols[i]:
        fig = go.Figure(
            go.Scatter(
                x=y.index, y=y.values,
                mode="lines+markers",
                name=target,
            )
        )
        fig.update_layout(
            title=target,
            height=220,
            margin=dict(l=10, r=10, t=40, b=30),
            showlegend=False,
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

st.success(f"Showing rows {start+1}-{end}.  Auto-advance every {refresh_rate}s ✔️")
