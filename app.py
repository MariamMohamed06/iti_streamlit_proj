"""
app.py — Job Change Predictor
==============================
A colorful Streamlit app that serves predictions from EVERY model trained
in ITI_Project_V2_.ipynb: KNN, LR + SMOTE, FE LR + SMOTE, Baseline Random
Forest, Tuned Random Forest, XGBoost + scale_pos_weight, XGBoost + SMOTE.

Run `train_all_models.py` first to produce the `artifacts/` folder this
app reads from, then:

    streamlit run app.py
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config + global style
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MODEL_COLORS = {
    "KNN": "#00C9A7",
    "LR + SMOTE": "#845EC2",
    "FE LR + SMOTE": "#D65DB1",
    "Baseline Random Forest": "#FF6F91",
    "Tuned Random Forest": "#FF9671",
    "XGBoost + scale_pos_weight": "#FFC75F",
    "XGBoost + SMOTE": "#F9F871",
}

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f1147 0%, #3a1c71 100%);
    }
    section[data-testid="stSidebar"] * { color: #f4f1ff !important; }

    h1, h2, h3, h4, .stMarkdown p { color: #f4f1ff; }

    .hero {
        background: linear-gradient(120deg, #7F00FF 0%, #E100FF 50%, #FF6F91 100%);
        padding: 28px 34px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(126, 0, 255, 0.35);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p { margin: 6px 0 0 0; color: #f0e6ff; font-size: 1.02rem; }

    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 22px 24px;
        backdrop-filter: blur(6px);
        margin-bottom: 18px;
    }

    .verdict-advance {
        background: linear-gradient(120deg, #11998e, #38ef7d);
        padding: 18px 22px; border-radius: 16px; text-align: center;
        font-size: 1.3rem; font-weight: 700; color: #05261d;
        box-shadow: 0 8px 24px rgba(17, 153, 142, 0.4);
    }
    .verdict-stay {
        background: linear-gradient(120deg, #ee0979, #ff6a00);
        padding: 18px 22px; border-radius: 16px; text-align: center;
        font-size: 1.3rem; font-weight: 700; color: #2b0400;
        box-shadow: 0 8px 24px rgba(238, 9, 121, 0.4);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 10px 14px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stButton > button {
        background: linear-gradient(120deg, #7F00FF, #E100FF);
        color: white; border: none; border-radius: 12px;
        padding: 0.6rem 1.4rem; font-weight: 700; font-size: 1rem;
        box-shadow: 0 6px 18px rgba(126, 0, 255, 0.45);
        transition: transform 0.15s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
def sanitize_columns(cols):
    clean = []
    for c in cols:
        c = c.replace("<", "lt").replace(">", "gt")
        c = re.sub(r"[\[\]{}():,]", "_", c)
        clean.append(c)
    return clean


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    if not ARTIFACTS_DIR.exists():
        return None

    encoder = joblib.load(ARTIFACTS_DIR / "encoder.pkl")
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        meta = json.load(f)

    models = {}
    for name, info in meta["models"].items():
        models[name] = joblib.load(ARTIFACTS_DIR / info["file"])

    return {"encoder": encoder, "meta": meta, "models": models}


def build_features(raw_df, feature_type, meta, encoder):
    cat_encoded = pd.DataFrame(
        encoder.transform(raw_df[meta["categorical_columns"]]),
        columns=encoder.get_feature_names_out(meta["categorical_columns"]),
        index=raw_df.index,
    )

    if feature_type == "fe":
        exp_num = raw_df["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)
        num = raw_df[meta["numerical_columns"]].copy()
        num["experience_to_training_ratio"] = exp_num / (raw_df["training_hours"] + 1)
        num["has_relevant_degree"] = (raw_df["major_discipline"] == "STEM").astype(int)
        full = pd.concat([cat_encoded, num], axis=1)
    else:
        full = pd.concat([cat_encoded, raw_df[meta["numerical_columns"]]], axis=1)

    full.columns = full.columns.astype(str)
    if feature_type == "xgb":
        full.columns = sanitize_columns(full.columns)

    order = meta["feature_orders"][feature_type]
    full = full.reindex(columns=order, fill_value=0)
    return full


def predict_all(raw_df, artifacts):
    meta = artifacts["meta"]
    encoder = artifacts["encoder"]
    results = {}
    for name, model in artifacts["models"].items():
        info = meta["models"][name]
        feats = build_features(raw_df, info["feature_type"], meta, encoder)
        proba = float(model.predict_proba(feats)[0][1])
        results[name] = {
            "probability": proba,
            "should_advance": proba >= info["threshold"],
            "threshold": info["threshold"],
        }
    return results


def gauge_chart(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 46, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(56, 239, 125, 0.25)"},
                {"range": [40, 70], "color": "rgba(255, 199, 95, 0.25)"},
                {"range": [70, 100], "color": "rgba(255, 90, 90, 0.30)"},
            ],
        },
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white", "family": "Arial"},
    )
    return fig


def comparison_chart(results):
    names = list(results.keys())
    probs = [results[n]["probability"] * 100 for n in names]
    order = np.argsort(probs)
    names = [names[i] for i in order]
    probs = [probs[i] for i in order]
    colors = [MODEL_COLORS.get(n, "#8888ff") for n in names]

    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"{p:.1f}%" for p in probs], textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=40, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], title="Probability of job change (%)",
                    color="white", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(color="white"),
        font=dict(color="white"),
    )
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎯 Job Change Predictor</h1>
        <p>Will a candidate look for a new job? Fill in their profile and get a live probability from every model in the notebook.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "No trained models found. Run **`python train_all_models.py`** "
        "(with `aug_train.csv` in the same folder) first — it creates an "
        "`artifacts/` folder next to this app with every model + the shared encoder."
    )
    st.stop()

meta = artifacts["meta"]
encoder = artifacts["encoder"]
cat_cols = meta["categorical_columns"]
cat_options = {col: list(cats) for col, cats in zip(cat_cols, encoder.categories_)}
num_ranges = meta["numeric_ranges"]

# --------------------------------------------------------------------------
# Sidebar — mode + leaderboard
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio(
        "Prediction mode",
        ["🔍 Single model", "📊 Compare all models"],
        index=1,
    )

    selected_model = None
    if mode == "🔍 Single model":
        model_names = list(meta["models"].keys())
        default_idx = model_names.index(meta["best_model_name"]) if meta["best_model_name"] in model_names else 0
        selected_model = st.selectbox("Choose a model", model_names, index=default_idx)

    st.markdown("---")
    st.markdown("## 🏆 Model Leaderboard")
    leaderboard = pd.DataFrame({
        name: info["metrics"] for name, info in meta["models"].items()
    }).T.sort_values("F1-score", ascending=False)
    leaderboard = leaderboard[["Accuracy", "Precision", "Recall", "F1-score"]].round(3)
    st.dataframe(leaderboard, use_container_width=True)
    st.caption(f"⭐ Best by F1-score: **{meta['best_model_name']}**")

# --------------------------------------------------------------------------
# Candidate form
# --------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🧑‍💼 Candidate profile")

c1, c2, c3 = st.columns(3)
with c1:
    city = st.selectbox("City", cat_options["city"])
    gender = st.selectbox("Gender", cat_options["gender"])
    relevent_experience = st.selectbox("Relevant experience", cat_options["relevent_experience"])
    enrolled_university = st.selectbox("Enrolled university", cat_options["enrolled_university"])
with c2:
    education_level = st.selectbox("Education level", cat_options["education_level"])
    major_discipline = st.selectbox("Major discipline", cat_options["major_discipline"])
    experience = st.selectbox("Years of experience", cat_options["experience"])
    last_new_job = st.selectbox("Years since last job change", cat_options["last_new_job"])
with c3:
    company_size = st.selectbox("Company size", cat_options["company_size"])
    company_type = st.selectbox("Company type", cat_options["company_type"])
    city_development_index = st.slider(
        "City development index",
        min_value=round(num_ranges["city_development_index"]["min"], 3),
        max_value=round(num_ranges["city_development_index"]["max"], 3),
        value=round(num_ranges["city_development_index"]["default"], 3),
        step=0.001,
    )
    training_hours = st.number_input(
        "Training hours",
        min_value=float(num_ranges["training_hours"]["min"]),
        max_value=float(num_ranges["training_hours"]["max"]) * 2,
        value=float(num_ranges["training_hours"]["default"]),
        step=1.0,
    )

predict_clicked = st.button("✨ Predict", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Prediction + results
# --------------------------------------------------------------------------
if predict_clicked:
    raw = pd.DataFrame([{
        "city": city,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "city_development_index": city_development_index,
        "training_hours": training_hours,
    }])

    results = predict_all(raw, artifacts)

    if mode == "🔍 Single model":
        info = results[selected_model]
        prob = info["probability"]
        color = MODEL_COLORS.get(selected_model, "#845EC2")

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(gauge_chart(prob, color), use_container_width=True)
        with right:
            st.markdown("<br>", unsafe_allow_html=True)
            if info["should_advance"]:
                st.markdown(
                    f'<div class="verdict-advance">🚀 Likely to look for a new job<br>'
                    f'<span style="font-size:0.95rem;font-weight:500">probability {prob:.1%} '
                    f'(threshold {info["threshold"]:.2f})</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-stay">🏠 Likely to stay<br>'
                    f'<span style="font-size:0.95rem;font-weight:500">probability {prob:.1%} '
                    f'(threshold {info["threshold"]:.2f})</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            m = meta["models"][selected_model]["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m['Accuracy']:.3f}")
            mc2.metric("Precision", f"{m['Precision']:.3f}")
            mc3.metric("Recall", f"{m['Recall']:.3f}")
            mc4.metric("F1-score", f"{m['F1-score']:.3f}")
            st.caption(f"Model used: **{selected_model}**")

    else:
        st.markdown("### 📊 Every model's prediction for this candidate")
        st.plotly_chart(comparison_chart(results), use_container_width=True)

        avg_prob = np.mean([r["probability"] for r in results.values()])
        votes_advance = sum(r["should_advance"] for r in results.values())
        total = len(results)

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Average probability", f"{avg_prob:.1%}")
        vc2.metric("Models voting 'advance'", f"{votes_advance} / {total}")
        vc3.metric("Consensus", "🚀 Advance" if votes_advance >= total / 2 else "🏠 Stay")

        table = pd.DataFrame({
            "Model": list(results.keys()),
            "Probability": [f"{r['probability']:.1%}" for r in results.values()],
            "Verdict": ["🚀 Advance" if r["should_advance"] else "🏠 Stay" for r in results.values()],
        }).sort_values("Probability", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("Fill in the candidate profile above and click **Predict** to see results.")
