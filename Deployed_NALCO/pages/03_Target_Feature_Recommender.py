##############################################################################
# pages/03_target_feature_recommender.py
#
# Recommend the top-3 process features to adjust in order to reach a user-
# chosen target (Conductivity, Elongation, or UTS).
##############################################################################
import os, json
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ──────────────────────────── CONFIG DEFAULTS ─────────────────────────────
DATA_FILE_DEFAULT    = "Training_Dataset.csv"   # 21-feature CSV

MODEL_DEFAULTS = {
    "Conductivity": "xgboost_model_output_Conductivity.pkl",
    "Elongation":   "xgboost_model_output_Elongation.pkl",
    "UTS":          "xgboost_model_output_   UTS.pkl",   # note spaces
}

# Which three levers matter most for each target (domain expert input)
TOP_FEATURES_MAP: Dict[str, List[str]] = {
    "Conductivity": [
        "STANDS_OIL_L_PR_VAL0",
        "QUENCH_CW_FLOW_EXIT_VAL0",
        "EMULSION_LEVEL_ANALO_VAL0",
    ],
    "Elongation": [
        "STANDS_OIL_L_PR_VAL0",
        "QUENCH_CW_FLOW_EXIT_VAL0",
        "QUENCH_CW_FLOW_ENTRY_VAL0",
    ],
    "UTS": [
        "STANDS_OIL_L_PR_VAL0",
        "QUENCH_CW_FLOW_EXIT_VAL0",
        "QUENCH_CW_FLOW_ENTRY_VAL0",
    ],
}

# Column names *inside the CSV* (UTS has leading spaces)
TARGET_TO_CSV_COL = {
    "Conductivity": "Conductivity",
    "Elongation":   "Elongation",
    "UTS":          "   UTS",
}

EPS = 1e-12          # avoid divide-by-zero in min-max scaling

# ────────────────────────── STREAMLIT PAGE CONFIG ─────────────────────────
st.set_page_config(page_title="Target Feature Recommender", layout="centered")
st.title("Target Feature Recommender")
st.markdown(
    "Set your *current* and *desired* property values below. "
    "When you click **Update Features**, the app identifies the **top three "
    "process levers** and shows their *initial* and *recommended* settings "
    "to help you reach the target."
)

# ───────────────────────────── SIDEBAR INPUTS ─────────────────────────────
# st.sidebar.header("Paths")

# data_path = st.sidebar.text_input("Predicted-data CSV", DATA_FILE_DEFAULT)
data_path = DATA_FILE_DEFAULT

# st.sidebar.markdown("**Model files:**")
model_paths: Dict[str, str] = {}
for tgt, default in MODEL_DEFAULTS.items():
    model_paths[tgt] = default

# ───────────────────────────── CACHED LOADERS ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        st.error(f"❌ CSV not found: {csv_path}"); st.stop()
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore")
    return df

@st.cache_resource(show_spinner=False)
def load_xgb(path: str):
    if not os.path.isfile(path):
        st.error(f"❌ Model not found: {path}"); st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

# ──────────────────── GENERIC PREDICT-ORDERED WRAPPER ─────────────────────
def predict_ordered(model, row: pd.Series | pd.DataFrame) -> np.ndarray:
    if hasattr(model, "get_booster"):
        exp_cols = list(model.get_booster().feature_names)
    else:
        exp_cols = list(model.feature_names_in_)
    if isinstance(row, pd.Series):
        row = row.to_frame().T
    for col in exp_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[exp_cols]
    return model.predict(row)

# ───────────────────── GRADIENT-DESCENT MINI-OPTIMISER ────────────────────
def gradient_optimize(
    model, start_norm: pd.Series, target_norm: float, features: Sequence[str],
    lr: float = 0.01, max_iter: int = 300, tol: float = 1e-3
) -> pd.Series:
    x = start_norm.copy()
    for _ in range(max_iter):
        pred = predict_ordered(model, x)[0]
        err  = target_norm - pred
        if abs(err) <= tol:
            break
        grads: Dict[str, float] = {}
        for f in features:
            plus, minus = x.copy(), x.copy()
            plus[f]  += 0.1
            minus[f] -= 0.1
            grads[f] = (predict_ordered(model, plus)[0] -
                        predict_ordered(model, minus)[0]) / 0.2
        for f, g in grads.items():
            x[f] += lr * g * err
            x[f]  = np.clip(x[f], 0.0, 1.0)
    return x

# ──────────────────────────── LOAD DATA/MODELS ────────────────────────────
df = load_dataset(data_path)

models = {tgt: load_xgb(path) for tgt, path in model_paths.items()}

# Ensure every expected column exists in df (add zeros then fill with median)
for tgt, mdl in models.items():
    exp_cols = (mdl.get_booster().feature_names
                if hasattr(mdl, "get_booster") else mdl.feature_names_in_)
    for col in exp_cols:
        if col not in df.columns:
            df[col] = 0.0
for col in df.columns:
    if df[col].nunique(dropna=True) == 1:         # constant / missing col
        df[col] = df[col].fillna(0)

# Pre-compute min / max and normalised frames
min_vals, max_vals = df.min(), df.max()
df_norm = (df - min_vals) / (max_vals - min_vals + EPS)

# Split once
output_cols = list(TARGET_TO_CSV_COL.values())
inputs_norm  = df_norm.drop(columns=output_cols)
outputs_norm = df_norm[output_cols]

def norm(v, col):   return (v - min_vals[col]) / (max_vals[col] - min_vals[col] + EPS)
def denorm(v, col): return v * (max_vals[col] - min_vals[col] + EPS) + min_vals[col]

# ────────────────────────────── UI SELECTION ──────────────────────────────
target_display = st.selectbox("Choose target to optimise:",
                              ["Conductivity", "Elongation", "UTS"])

csv_col   = TARGET_TO_CSV_COL[target_display]
model     = models[target_display]
top_feats = TOP_FEATURES_MAP[target_display]

t_min, t_max = float(min_vals[csv_col]), float(max_vals[csv_col])
cur_val = st.slider(f"Current {target_display}", min_value=t_min, max_value=t_max,
                    value=float((t_min + t_max) / 2), format="%.5f")
des_val = st.slider(f"Desired {target_display}", min_value=t_min, max_value=t_max,
                    value=min(cur_val + 0.1, t_max), format="%.5f")

# ───────────────────────────── PREDICT BUTTON ─────────────────────────────
if st.button("🔄 Update Features", use_container_width=True):

    cur_norm = norm(cur_val, csv_col)
    des_norm = norm(des_val, csv_col)

    # nearest existing row as starting point
    idx0 = (outputs_norm[csv_col] - cur_norm).abs().idxmin()
    start_row_norm = inputs_norm.iloc[idx0].copy()

    # gradient optimisation
    opt_norm = gradient_optimize(model, start_row_norm, des_norm, top_feats)

    # to physical units
    init_phys = {f: denorm(start_row_norm[f], f) for f in top_feats}
    new_phys  = {f: denorm(opt_norm[f],        f) for f in top_feats}

    res_df = pd.DataFrame({
        "Feature": top_feats,
        "Initial Value": [round(init_phys[f], 5) for f in top_feats],
        "Updated Value": [round(new_phys[f], 5)  for f in top_feats],
    })

    st.subheader(f"Recommended Adjustments for {target_display}")
    st.dataframe(res_df, hide_index=True, use_container_width=True)

    json_out = json.dumps({row["Feature"]: row["Updated Value"]
                           for _, row in res_df.iterrows()}, indent=2)
    st.download_button("📥 Download Updated Values",
                       data=json_out,
                       file_name=f"{target_display.lower()}_tweaks.json",
                       mime="application/json")
