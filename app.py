# app.py - Machine Failure Predictor (Streamlit app)
# Run it with:  streamlit run app.py
# It needs these files next to it (the notebook creates them):
#   models/best_model.joblib, results/best_per_model.csv, figs/*.png

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Machine Failure Predictor", layout="wide")


# load the saved model once (the file has the pipeline and a few extra settings)
@st.cache_resource
def load_bundle():
    return joblib.load(HERE / "models" / "best_model.joblib")


bundle = load_bundle()
pipe = bundle["pipeline"]
FEATURES = bundle["features"]
BASE_COLS = bundle["base_cols"]
RANGES = bundle["ranges"]
PRESETS = bundle["presets"]
DEFAULT_THR = float(min(max(bundle["threshold"], 0.01), 0.99))

LABELS = {
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


# same feature steps as the notebook (must match exactly)
def make_features(d):
    out = d[BASE_COLS].copy()
    out["temp_diff"] = d["process_temp"] - d["air_temp"]
    out["power"] = d["torque"] * d["rpm"] * 2 * np.pi / 60
    out["wear_torque"] = d["tool_wear"] * d["torque"]
    out["type_L"] = (d["type"] == "L").astype(int)
    out["type_M"] = (d["type"] == "M").astype(int)
    return out[FEATURES]


# failure probability for a table of machines
def failure_probability(d):
    return pipe.predict_proba(make_features(d))[:, 1]


# button action: put the values of a ready-made example into the sliders
def apply_preset(name):
    p = PRESETS.get(name)
    if p:
        for c in BASE_COLS:
            st.session_state[c] = float(p[c])
        st.session_state["type"] = p["type"]


# give every slider a starting value the first time the app opens
def init_state():
    start = PRESETS.get("Healthy machine")
    for c in BASE_COLS:
        if c not in st.session_state:
            lo, hi = RANGES[c]
            st.session_state[c] = float(start[c]) if start else float((lo + hi) / 2)
    if "type" not in st.session_state:
        st.session_state["type"] = start["type"] if start else "L"
    if "thr" not in st.session_state:
        st.session_state["thr"] = round(DEFAULT_THR, 2)


# ---------------------------------------------------------------- pages
def page_predict():
    init_state()
    st.title("Machine Failure Predictor")
    st.caption(
        f"Model: {bundle['model_name']} ({bundle['imbalance']}). "
        "Move the sliders and watch the alert change."
    )

    st.write("Try a ready-made example:")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    left, right = st.columns([3, 2])

    with left:
        values = {}
        for c in BASE_COLS:
            lo, hi = RANGES[c]
            step = 1.0 if c in ("rpm", "tool_wear") else 0.1
            values[c] = st.slider(LABELS[c], float(lo), float(hi), step=step, key=c)
        values["type"] = st.selectbox(
            "Product quality type", ["L", "M", "H"], key="type"
        )
        thr = st.slider(
            "Alert threshold (lower = more alerts, fewer missed failures)",
            0.01, 0.99, step=0.01, key="thr",
        )

    row = pd.DataFrame([values])
    proba = float(failure_probability(row)[0])
    feats = make_features(row).iloc[0]

    with right:
        st.subheader("Result")
        if proba >= thr:
            st.error("Machine will fail!")
        else:
            st.success("Machine is healthy")
        st.metric("Failure probability", f"{proba * 100:.1f} %")
        st.progress(min(max(proba, 0.0), 1.0))

        st.subheader("Why this reading looks risky")
        st.caption("Simple rule-of-thumb checks from the dataset description.")
        notes = []
        limit = {"L": 11000, "M": 12000, "H": 13000}[values["type"]]
        if feats["temp_diff"] < 8.6 and values["rpm"] < 1380:
            notes.append(
                f"Heat: temperature gap is only {feats['temp_diff']:.1f} K and the "
                "speed is low."
            )
        if feats["power"] < 3500 or feats["power"] > 9000:
            notes.append(
                f"Power is {feats['power']:.0f} W, outside the normal 3500 to 9000 W."
            )
        if feats["wear_torque"] > limit:
            notes.append(
                f"Overstrain: tool wear x torque is {feats['wear_torque']:.0f}, "
                f"above {limit} for type {values['type']}."
            )
        if values["tool_wear"] >= 200:
            notes.append("The tool is heavily worn (200 minutes or more).")
        if notes:
            for n in notes:
                st.warning(n)
        else:
            st.info("No rule-of-thumb warning for these readings.")

        with st.expander("Derived values"):
            st.write(
                {
                    "temperature gap [K]": round(float(feats["temp_diff"]), 2),
                    "power [W]": round(float(feats["power"]), 1),
                    "tool wear x torque": round(float(feats["wear_torque"]), 1),
                }
            )


def show_figure(file_name, caption):
    path = HERE / "figs" / file_name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{file_name} not found. Run the notebook first.")


def page_models():
    st.title("Model comparison")
    st.write(
        "All models use the same 80/20 split. The test set was used only for the "
        "final scores."
    )
    table = HERE / "results" / "best_per_model.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table).round(3), use_container_width=True)
    else:
        st.info("results/best_per_model.csv not found. Run the notebook first.")

    figs = {
        "Before vs after tuning": "08_before_after_tuning.png",
        "Test scores of each model": "09_test_scores.png",
        "Confusion matrices": "10_confusion_matrices.png",
        "Precision-Recall curves": "11_pr_curves.png",
        "Feature importance (Random Forest)": "12_feature_importance.png",
        "Effect of tree depth and k": "13_setting_curves.png",
        "Alert threshold trade-off": "14_threshold.png",
        "Small decision tree": "15_decision_tree.png",
    }
    choice = st.selectbox("Choose a chart", list(figs))
    show_figure(figs[choice], choice)


def page_insights():
    st.title("Data insights")
    figs = [
        ("Failure vs no failure", "01_failure_count.png"),
        ("Readings for fine and failed machines", "02_histograms.png"),
        ("Box plots", "03_boxplots.png"),
        ("Correlation heatmap", "04_correlation.png"),
        ("Failure rate by product type", "05_failure_by_type.png"),
        ("Torque vs speed", "06_torque_rpm.png"),
        ("New features", "07_new_features.png"),
    ]
    a, b = st.columns(2)
    for i, (title, file_name) in enumerate(figs):
        with (a if i % 2 == 0 else b):
            st.subheader(title)
            show_figure(file_name, title)


def page_batch():
    st.title("Batch check")
    st.write(
        "Upload a CSV with the columns Type, Air temperature [K], Process "
        "temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]. "
        "The simple names (type, air_temp, ...) also work."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is None:
        return
    data = pd.read_csv(file).rename(
        columns={
            "Type": "type",
            "Air temperature [K]": "air_temp",
            "Process temperature [K]": "process_temp",
            "Rotational speed [rpm]": "rpm",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
        }
    )
    need = ["type"] + BASE_COLS
    missing = [c for c in need if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    data["failure_probability"] = failure_probability(data)
    data["alert"] = np.where(
        data["failure_probability"] >= DEFAULT_THR, "Machine will fail!", "Healthy"
    )
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Download predictions",
        data.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
    )


# ---------------------------------------------------------------- menu
page = st.sidebar.radio(
    "Menu", ["Predict", "Model comparison", "Data insights", "Batch check"]
)
if page == "Predict":
    page_predict()
elif page == "Model comparison":
    page_models()
elif page == "Data insights":
    page_insights()
else:
    page_batch()
