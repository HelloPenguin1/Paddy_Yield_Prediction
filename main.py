from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_best.pkl"

FEATURE_ORDER = [
    "Micronutrients_70Days",
    "LP_Mainfield(in Tonnes)",
    "DAP_20days",
    "LP_nurseryarea(in Tonnes)",
    "Seedrate(in Kg)",
    "Nursery area (Cents)",
    "Pest_60Day(in ml)",
    "Weed28D_thiobencarb",
    "Urea_40Days",
    "Potassh_50Days",
]

# Mean and std (ddof=0) computed from the full raw dataset.
# LP_Mainfield, Urea_40Days, Potassh_50Days were StandardScaler-transformed
# during preprocessing; the remaining 7 integer features were kept as-is.
SCALER_PARAMS = {
    "LP_Mainfield(in Tonnes)": {"mean": 46.4682681965, "std": 17.9689878831},
    "Urea_40Days":             {"mean": 100.8547292937, "std": 38.9998913014},
    "Potassh_50Days":          {"mean": 38.5872499104, "std": 14.9214475381},
}

FEATURE_META = {
    "Seedrate(in Kg)": {
        "desc": "Seeding density: amount of seed used per unit area",
        "min": 25.0, "max": 150.0, "step": 5.0, "default": 100.0,
    },
    "Nursery area (Cents)": {
        "desc": "Nursery land area in cents (1 cent ≈ 40.5 m²)",
        "min": 20.0, "max": 120.0, "step": 5.0, "default": 80.0,
    },
    "LP_nurseryarea(in Tonnes)": {
        "desc": "Land-preparation manure applied to the nursery",
        "min": 1.0, "max": 6.0, "step": 1.0, "default": 4.0,
    },
    "LP_Mainfield(in Tonnes)": {
        "desc": "Land-preparation manure applied to the main field",
        "min": 12.5, "max": 75.0, "step": 2.5, "default": 50.0,
    },
    "DAP_20days": {
        "desc": "DAP (Di-Ammonium Phosphate) fertiliser applied at 20 days (kg)",
        "min": 40.0, "max": 240.0, "step": 10.0, "default": 160.0,
    },
    "Weed28D_thiobencarb": {
        "desc": "Thiobencarb weedicide applied at 28 days (kg)",
        "min": 2.0, "max": 12.0, "step": 1.0, "default": 8.0,
    },
    "Urea_40Days": {
        "desc": "Urea fertiliser applied at 40 days (kg)",
        "min": 27.13, "max": 162.78, "step": 1.0, "default": 100.0,
    },
    "Potassh_50Days": {
        "desc": "Potash fertiliser applied at 50 days (kg)",
        "min": 10.38, "max": 62.28, "step": 1.0, "default": 38.0,
    },
    "Micronutrients_70Days": {
        "desc": "Micronutrients applied at 70 days (kg)",
        "min": 15.0, "max": 90.0, "step": 5.0, "default": 60.0,
    },
    "Pest_60Day(in ml)": {
        "desc": "Pesticide applied at 60 days (ml)",
        "min": 600.0, "max": 3600.0, "step": 100.0, "default": 2400.0,
    },
}

INPUT_ORDER = [
    "Seedrate(in Kg)",
    "Nursery area (Cents)",
    "LP_nurseryarea(in Tonnes)",
    "LP_Mainfield(in Tonnes)",
    "DAP_20days",
    "Weed28D_thiobencarb",
    "Urea_40Days",
    "Potassh_50Days",
    "Micronutrients_70Days",
    "Pest_60Day(in ml)",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def standardize(value: float, feature: str) -> float:
    if feature in SCALER_PARAMS:
        p = SCALER_PARAMS[feature]
        return (value - p["mean"]) / p["std"]
    return value


def main() -> None:
    st.set_page_config(page_title="Paddy Yield Predictor", page_icon="🌾", layout="centered")

    st.title("Paddy Yield Predictor")
    st.markdown(
        "Predict total paddy yield (kg) using a **Random Forest** model trained on the "
        "[UCI Paddy Dataset](https://datalab-12.ics.uci.edu/dataset/1186/paddy+dataset) "
        "from Tamil Nadu, India. Fill in the 10 agronomic inputs below and click **Predict**."
    )

    st.divider()

    model = load_model()
    raw_inputs: dict[str, float] = {}

    col_left, col_right = st.columns(2)

    for idx, feature in enumerate(INPUT_ORDER):
        meta = FEATURE_META[feature]
        target_col = col_left if idx % 2 == 0 else col_right

        with target_col:
            st.caption(f"**{feature}**: {meta['desc']}")
            st.caption(f"Dataset range: **{meta['min']}** - **{meta['max']}**")
            raw_inputs[feature] = st.number_input(
                label=feature,
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                label_visibility="collapsed",
            )

    st.divider()

    if st.button("Predict Yield", type="primary", use_container_width=True):
        row = {f: standardize(raw_inputs[f], f) for f in FEATURE_ORDER}
        X = pd.DataFrame([row], columns=FEATURE_ORDER)
        prediction = model.predict(X)[0]

        st.success(f"Predicted Paddy Yield: **{prediction:,.0f} kg**")

        with st.expander("Model input details"):
            st.dataframe(
                pd.DataFrame([raw_inputs], columns=INPUT_ORDER),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
