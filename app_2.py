"""
app.py — Recruitment Screening Dashboard
=========================================
A professional Streamlit dashboard that serves live predictions from every
model trained in ITI_Project_V2_.ipynb: KNN, LR + SMOTE, FE LR + SMOTE,
Baseline Random Forest, Tuned Random Forest, XGBoost + scale_pos_weight,
XGBoost + SMOTE.

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
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Recruitment Screening Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# A single muted, professional blue/slate scale shared across every chart —
# darker = better-ranked model. Assigned once the leaderboard is known.
PALETTE = ["#0B3C7A", "#1D5FA6", "#2C7DC2", "#4A9BD1", "#7DB8DD", "#A9CFE6", "#CFE3F0"]

ACCENT = "#1D5FA6"
SUCCESS = "#1E7A46"
DANGER = "#B3312C"
INK = "#1F2937"
MUTED = "#5B6472"
BORDER = "#E2E5EA"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F4F6F9"

CUSTOM_CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    }}
    .stApp {{ background: {PAGE_BG}; }}

    section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: {INK};
        font-weight: 600;
    }}

    h1, h2, h3, h4 {{ color: {INK}; font-weight: 650; }}
    p, span, label, .stMarkdown {{ color: {INK}; }}
    .subtle {{ color: {MUTED}; }}

    .app-header {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-left: 5px solid {ACCENT};
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 22px;
    }}
    .app-header h1 {{ margin: 0; font-size: 1.7rem; }}
    .app-header p {{ margin: 6px 0 0 0; color: {MUTED}; font-size: 0.98rem; }}

    .card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }}
    .card h3 {{ margin-top: 0; font-size: 1.05rem; }}

    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-best {{ background: #E7F1EA; color: {SUCCESS}; }}

    .verdict {{
        border-radius: 10px;
        padding: 18px 22px;
        border: 1px solid {BORDER};
        border-left: 5px solid {ACCENT};
    }}
    .verdict-advance {{ border-left-color: {SUCCESS}; }}
    .verdict-stay {{ border-left-color: {DANGER}; }}
    .verdict .label {{
        font-size: 1.05rem; font-weight: 700; color: {INK}; margin: 0 0 2px 0;
    }}
    .verdict .sub {{ font-size: 0.88rem; color: {MUTED}; margin: 0; }}

    div[data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}

    .stButton > button {{
        background: {ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .stButton > button:hover {{ background: #16497F; }}

    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

    hr {{ border-color: {BORDER}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
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


def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 11}},
            "bar": {"color": ACCENT, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 40], "color": "#EAF2E9"},
                {"range": [40, 70], "color": "#FBF3DE"},
                {"range": [70, 100], "color": "#F7E6E4"},
            ],
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": INK, "family": "Segoe UI"},
    )
    return fig


def comparison_chart(results, ranking_order):
    # Colour by leaderboard rank (best model = darkest), independent of the
    # probability value, so the chart reads the same way every time.
    color_map = {name: PALETTE[min(i, len(PALETTE) - 1)] for i, name in enumerate(ranking_order)}

    names = list(results.keys())
    probs = [results[n]["probability"] * 100 for n in names]
    order = np.argsort(probs)
    names = [names[i] for i in order]
    probs = [probs[i] for i in order]
    colors = [color_map.get(n, MUTED) for n in names]

    fig = go.Figure(go.Bar(
        x=probs, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color=BORDER, width=1)),
        text=[f"{p:.1f}%" for p in probs], textposition="outside",
        textfont=dict(color=INK, size=12),
    ))
    fig.update_layout(
        height=400, margin=dict(l=10, r=40, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 108], title="Probability of job change (%)",
                    color=MUTED, gridcolor=BORDER, zeroline=False),
        yaxis=dict(color=INK),
        font=dict(color=INK, family="Segoe UI"),
    )
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Recruitment Screening Dashboard</h1>
        <p>Estimate the probability that a candidate will look for a new job, using every model trained in the notebook.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "No trained models found. Run **`python train_all_models.py`** "
        "(with `aug_train.csv` in the same folder) first — it creates an "
        "`artifacts/` folder next to this app with every model and the shared encoder."
    )
    st.stop()

meta = artifacts["meta"]
encoder = artifacts["encoder"]
cat_cols = meta["categorical_columns"]
cat_options = {col: list(cats) for col, cats in zip(cat_cols, encoder.categories_)}
num_ranges = meta["numeric_ranges"]

leaderboard_full = pd.DataFrame({
    name: info["metrics"] for name, info in meta["models"].items()
}).T.sort_values("F1-score", ascending=False)
ranking_order = list(leaderboard_full.index)

# --------------------------------------------------------------------------
# Sidebar — mode + leaderboard
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    mode = st.radio(
        "Prediction mode",
        ["Single model", "Compare all models"],
        index=1,
    )

    selected_model = None
    if mode == "Single model":
        model_names = list(meta["models"].keys())
        default_idx = model_names.index(meta["best_model_name"]) if meta["best_model_name"] in model_names else 0
        selected_model = st.selectbox("Choose a model", model_names, index=default_idx)

    st.markdown("---")
    st.markdown("### Model leaderboard")
    st.caption("Ranked by F1-score on the held-out test set")
    leaderboard = leaderboard_full[["Accuracy", "Precision", "Recall", "F1-score"]].round(3)
    st.dataframe(leaderboard, use_container_width=True)
    st.markdown(
        f'<span class="badge badge-best">Best model — {meta["best_model_name"]}</span>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Candidate form
# --------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Candidate profile")

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

predict_clicked = st.button("Run prediction", use_container_width=False)
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

    if mode == "Single model":
        info = results[selected_model]
        prob = info["probability"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)
        with right:
            st.markdown("<br>", unsafe_allow_html=True)
            if info["should_advance"]:
                st.markdown(
                    f'<div class="verdict verdict-advance">'
                    f'<p class="label">Recommended to advance</p>'
                    f'<p class="sub">Probability {prob:.1%} · threshold {info["threshold"]:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict verdict-stay">'
                    f'<p class="label">Not recommended to advance</p>'
                    f'<p class="sub">Probability {prob:.1%} · threshold {info["threshold"]:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            m = meta["models"][selected_model]["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m['Accuracy']:.3f}")
            mc2.metric("Precision", f"{m['Precision']:.3f}")
            mc3.metric("Recall", f"{m['Recall']:.3f}")
            mc4.metric("F1-score", f"{m['F1-score']:.3f}")
            st.caption(f"Model used: {selected_model}")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Model-by-model comparison")
        st.plotly_chart(comparison_chart(results, ranking_order), use_container_width=True)

        avg_prob = np.mean([r["probability"] for r in results.values()])
        votes_advance = sum(r["should_advance"] for r in results.values())
        total = len(results)

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Average probability", f"{avg_prob:.1%}")
        vc2.metric("Models recommending advance", f"{votes_advance} / {total}")
        vc3.metric("Consensus", "Advance" if votes_advance >= total / 2 else "Do not advance")

        table = pd.DataFrame({
            "Model": list(results.keys()),
            "Probability": [r["probability"] for r in results.values()],
            "Verdict": ["Advance" if r["should_advance"] else "Do not advance" for r in results.values()],
        }).sort_values("Probability", ascending=False)
        table["Probability"] = table["Probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Fill in the candidate profile above and click **Run prediction** to see results.")