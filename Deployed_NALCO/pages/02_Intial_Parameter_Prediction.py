##############################################################################
# pages/02_reverse_predictor.py  –  3 targets ➜ 16 process parameters
##############################################################################
import os, json, math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

# ─────────────────────────── CONFIGURATION ────────────────────────────────
st.set_page_config(page_title="Intial Parameter Predictor", layout="centered")

# ── file locations ────────────────────────────────────────────────────────
MODEL_DIR = ""         # leave empty if checkpoints live next to this script
CHECKPOINTS = {
    "gen_input":  "pages\gen_input.pth.tar",   # 3 → 16 generator
    "gen_output": "pages\gen_output.pth.tar",  # 16 → 3 generator (unused here)
    "disc_input": "pages\disc_input.pth.tar",  # (unused)
    "disc_output":"pages\disc_output.pth.tar", # (unused)
}

# ── column layout ─────────────────────────────────────────────────────────
INPUT_COLS  = ["UTS", "Elongation", "Conductivity"]      # slider names
RAW_INPUT_COLS = ["   UTS", "Elongation", "Conductivity"]# names in CSV/model
OUTPUT_COLS = [
    "EMUL_OIL_L_TEMP_PV_VAL0","STAND_OIL_L_TEMP_PV_REAL_VAL0",
    "GEAR_OIL_L_TEMP_PV_REAL_VAL0","EMUL_OIL_L_PR_VAL0",
    "QUENCH_CW_FLOW_EXIT_VAL0","CAST_WHEEL_RPM_VAL0",
    "BAR_TEMP_VAL0","QUENCH_CW_FLOW_ENTRY_VAL0","GEAR_OIL_L_PR_VAL0",
    "STANDS_OIL_L_PR_VAL0","TUNDISH_TEMP_VAL0","RM_MOTOR_COOL_WATER__VAL0",
    "ROLL_MILL_AMPS_VAL0","RM_COOL_WATER_FLOW_VAL0",
    "EMULSION_LEVEL_ANALO_VAL0","Furnace_Temperature",
]

DEVICE = "cpu"   # set "cuda" if you have a GPU on the server

# ────────────────────────────── SIDEBAR ───────────────────────────────────
# st.sidebar.header("Data / Checkpoints")

# train_csv_path = st.sidebar.text_input(
#     "Training CSV (contains full min–max stats)",
#     value="Training_Dataset.csv",
# )

train_csv_path = "Training_Dataset.csv"

# show checkpoint folder
# st.sidebar.write("**Checkpoint folder:**", os.path.abspath(MODEL_DIR or "."))

# ──────────────────────── HELPER CLASSES/FNS ──────────────────────────────
class Generator(nn.Module):
    """Exact architecture from your snippet – 3 hidden layers, InstanceNorm."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource(hash_funcs={torch.nn.Module: id}, show_spinner=False)
def load_rev_generator(path: str) -> nn.Module:
    """Load 3→16 generator with its checkpoint weights."""
    gen = Generator(input_dim=3, output_dim=16).to(DEVICE)
    if not os.path.isfile(path):
        st.error(f"❌ Checkpoint not found: `{path}`"); st.stop()
    ckpt = torch.load(path, map_location=torch.device(DEVICE))
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    return gen

@st.cache_data(show_spinner=False)
def load_training_stats(csv_path: str):
    """Read training CSV once and return min/max Series for all needed cols."""
    if not os.path.isfile(csv_path):
        st.error(f"❌ CSV not found at `{csv_path}`"); st.stop()
    df = pd.read_csv(csv_path)
    # drop chems & unnamed like in your training script
    df = df.drop(columns=[c for c in df.columns if c.strip() in
                          ["'%SI'","%SI","%FE","%TI","%V","%AL",'Unnamed: 0'] if c in df], errors="ignore")
    min_s = df[RAW_INPUT_COLS + OUTPUT_COLS].min()
    max_s = df[RAW_INPUT_COLS + OUTPUT_COLS].max()
    return min_s, max_s, df  # (df is optional; kept for extra stats)

def normalize(x, col, s_min, s_max):
    return (x - s_min[col]) / (s_max[col] - s_min[col] + 1e-12)

def denormalize(x, col, s_min, s_max):
    return x * (s_max[col] - s_min[col] + 1e-12) + s_min[col]

# ──────────────────────────── LOAD ASSETS ─────────────────────────────────
min_s, max_s, _train_df = load_training_stats(train_csv_path)
gen3to16 = load_rev_generator(os.path.join(MODEL_DIR, CHECKPOINTS["gen_input"]))

# ─────────────────────────── STREAMLIT UI ────────────────────────────────
st.title("Intial Parameter Predictor")

st.markdown(
    "Move the sliders to set your target **UTS**, **Elongation**, and "
    "**Conductivity**.  Click **Predict** and the Cycle-GAN generator will "
    "output the 16 upstream process parameters likely to achieve those targets."
)

slider_vals: Dict[str, float] = {}
for ui_col, raw_col in zip(INPUT_COLS, RAW_INPUT_COLS):
    slider_vals[raw_col] = st.slider(
        ui_col,
        min_value=float(min_s[raw_col]),
        max_value=float(max_s[raw_col]),
        value=float((min_s[raw_col] + max_s[raw_col]) / 2),
        step=float((max_s[raw_col] - min_s[raw_col]) / 500.0),
        format="%.3f",
    )

if st.button("🔮 Predict", use_container_width=True):
    # ── 1. normalise the three inputs
    inp_vec = torch.tensor(
        [normalize(slider_vals[c], c, min_s, max_s) for c in RAW_INPUT_COLS],
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    # ── 2. generator forward pass
    with torch.no_grad():
        pred_norm = gen3to16(inp_vec).cpu().squeeze(0).numpy()

    # ── 3. denormalise to physical units
    pred_phys = {
        col: float(denormalize(pred_norm[i], col, min_s, max_s))
        for i, col in enumerate(OUTPUT_COLS)
    }

    # ── 4. pretty display
    pred_df = (
        pd.DataFrame([pred_phys])
        .T.rename(columns={0: "Predicted Value"})
        .round(4)
    )
    st.subheader("Predicted Process Parameters")
    st.dataframe(pred_df, use_container_width=True)

    # ── 5. allow download as JSON
    json_out = json.dumps(pred_phys, indent=2)
    st.download_button(
        label="📥 Download JSON", data=json_out,
        file_name="Intial Parameter Predictor.json", mime="application/json"
    )
