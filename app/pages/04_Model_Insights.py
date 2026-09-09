import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="TeleMetric | Model Insights", page_icon="🧠", layout="wide")

# Custom CSS for the TeleMetric visual theme.
cosmic_theme = """
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 12, 41, 0.6);
}

.telemetric-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #e74c3c, #3498db, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
    padding-bottom: 10px;
}
</style>
"""

st.markdown(cosmic_theme, unsafe_allow_html=True)
st.markdown('<p class="telemetric-title">TeleMetric</p>', unsafe_allow_html=True)
st.markdown("### Customer Retention & Churn Intelligence")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_comparison.csv"
MODEL_EVALUATION_PATH = OUTPUT_DIR / "model_evaluation.csv"
PARAMS_PATH = OUTPUT_DIR / "best_model_params.json"
SHAP_IMAGE_PATH = OUTPUT_DIR / "shap_feature_importance.png"
CALIBRATION_IMAGE_PATH = OUTPUT_DIR / "xgboost_calibration_curve.png"

# -----------------------------------------------------------------------------
# Load saved evaluation artifacts
# -----------------------------------------------------------------------------
try:
    model_results = pd.read_csv(MODEL_RESULTS_PATH)
except FileNotFoundError:
    st.error("model_comparison.csv was not found inside the outputs folder.")
    st.stop()

try:
    evaluation_df = pd.read_csv(MODEL_EVALUATION_PATH)
except FileNotFoundError:
    evaluation_df = pd.DataFrame()

try:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    best_params = {}

st.title("Model Comparison & Explainability")
st.write(
    "Model benchmarking, the final XGBoost modeling path, probability calibration, "
    "and global SHAP explainability."
)

# -----------------------------------------------------------------------------
# Be explicit about the fact that these are different experiments.
# The original leaderboard contains untuned baseline models, while XGBoost was
# subsequently tuned in a separate RandomizedSearchCV experiment.
# -----------------------------------------------------------------------------
benchmark_best = model_results.loc[model_results["ROC_AUC"].idxmax()]
benchmark_model = benchmark_best["Model"]
benchmark_auc = benchmark_best["ROC_AUC"]
benchmark_std = benchmark_best.get("ROC_AUC_STD", float("nan"))

xgb_tuned_cv = 0.8484864803582564
xgb_test_auc = 0.8415147718860492
xgb_test_pr_auc = 0.6530851222334262

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        f"**Baseline CV leader**\n\n"
        f"### {benchmark_model}\n"
        f"ROC-AUC {benchmark_auc * 100:.2f}% · CV std {benchmark_std * 100:.2f} pp"
    )
with col2:
    st.info(
        "**Operational model**\n\n"
        "### Tuned XGBoost\n"
        f"Search CV ROC-AUC {xgb_tuned_cv * 100:.2f}%"
    )
with col3:
    st.info(
        "**Held-out test**\n\n"
        "### Tuned XGBoost\n"
        f"ROC-AUC {xgb_test_auc * 100:.2f}% · PR-AUC {xgb_test_pr_auc * 100:.2f}%"
    )

st.caption(
    "Important: the 84.73% Logistic Regression score is from the initial untuned "
    "5-fold comparison. The 84.85% XGBoost score is from a separate hyperparameter "
    "search, so those two numbers are not a like-for-like final model comparison."
)

st.divider()

# -----------------------------------------------------------------------------
# Baseline model comparison
# -----------------------------------------------------------------------------
st.subheader("1. Baseline Model Comparison")
st.write(
    "The initial experiment compared Logistic Regression, Random Forest and XGBoost "
    "using the same 5-fold StratifiedKFold setup and ROC-AUC as the scoring metric. "
    "This identifies the baseline ROC-AUC leader; it does not by itself establish the "
    "best final production model across every possible metric."
)

display_results = model_results.copy()
display_results["ROC_AUC"] = display_results["ROC_AUC"].round(3)
if "ROC_AUC_STD" in display_results.columns:
    display_results["ROC_AUC_STD"] = display_results["ROC_AUC_STD"].round(3)
display_results = display_results.sort_values("ROC_AUC", ascending=False)

st.dataframe(display_results, use_container_width=True, hide_index=True)
st.caption(
    "Baseline leaderboard: mean 5-fold ROC-AUC ± fold-to-fold standard deviation. "
    "Higher ROC-AUC is better; lower standard deviation indicates more stable fold scores."
)

if "ROC_AUC" in model_results.columns:
    st.subheader("Baseline ROC-AUC")
    fig_auc = px.bar(
        model_results,
        x="Model",
        y="ROC_AUC",
        text="ROC_AUC",
        color="Model",
        color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"],
    )
    fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_auc.update_layout(
        template="plotly_dark",
        yaxis_title="ROC-AUC",
        xaxis_title="",
        yaxis_range=[
            max(0, model_results["ROC_AUC"].min() - 0.05),
            min(1, model_results["ROC_AUC"].max() + 0.05),
        ],
        height=400,
    )
    st.plotly_chart(fig_auc, use_container_width=True)

if "ROC_AUC_STD" in model_results.columns:
    st.subheader("Cross-Validation Stability")
    fig_std = px.bar(
        model_results,
        x="Model",
        y="ROC_AUC_STD",
        text="ROC_AUC_STD",
        color="Model",
        color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"],
    )
    fig_std.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_std.update_layout(
        template="plotly_dark",
        yaxis_title="ROC-AUC Standard Deviation",
        xaxis_title="",
        height=400,
    )
    st.plotly_chart(fig_std, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Tuned XGBoost
# -----------------------------------------------------------------------------
st.subheader("2. Tuned XGBoost — Final Modeling Path")
st.write(
    "After the baseline comparison, XGBoost was tuned separately with "
    "RandomizedSearchCV using 20 parameter combinations and 5-fold cross-validation. "
    "The selected configuration was then evaluated on the held-out test set."
)

if best_params:
    params_df = pd.DataFrame(
        [{"Hyperparameter": key, "Selected value": value} for key, value in best_params.items()]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best search CV ROC-AUC", f"{xgb_tuned_cv:.4f}")
with col2:
    st.metric("Held-out test ROC-AUC", f"{xgb_test_auc:.4f}")
with col3:
    st.metric("Held-out test PR-AUC", f"{xgb_test_pr_auc:.4f}")

st.warning(
    "The tuned XGBoost search score (0.8485) should not be treated as a direct "
    "replacement for the 0.8473 Logistic Regression baseline score. Logistic "
    "Regression was not subjected to the same hyperparameter search. The held-out "
    "test scores are the more direct generalization check for the selected XGBoost model."
)

st.divider()

# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------
st.header("3. Model Explainability (SHAP)")
st.write(
    "SHAP (SHapley Additive exPlanations) estimates how much each feature contributes "
    "to a model prediction. The artifact shown here is a global importance view for "
    "the tree-based XGBoost model."
)

if SHAP_IMAGE_PATH.exists():
    try:
        img_shap = Image.open(SHAP_IMAGE_PATH)
        st.image(img_shap, caption="Global Feature Importance (TreeSHAP)", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load SHAP image: {e}")
else:
    st.warning("SHAP feature importance image was not found.")

st.caption(
    "This repository currently stores the SHAP feature-importance visualization as a PNG artifact."
)

st.divider()

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
st.header("4. Probability Calibration")
st.write(
    "Calibration is a post-processing step that attempts to make predicted churn "
    "probabilities better match observed churn frequencies. It is evaluated separately "
    "from model discrimination."
)

if CALIBRATION_IMAGE_PATH.exists():
    try:
        img_calib = Image.open(CALIBRATION_IMAGE_PATH)
        st.image(
            img_calib,
            caption="OOF Probability Calibration: Raw XGBoost vs Calibrated XGBoost",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Failed to load calibration image: {e}")
else:
    st.warning("xgboost_calibration_curve.png was not found.")

if not evaluation_df.empty:
    st.subheader("OOF Evaluation — Raw vs Calibrated XGBoost")
    st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

    raw = evaluation_df.loc[evaluation_df["Model"].eq("XGBoost (raw OOF)")]
    calibrated = evaluation_df.loc[evaluation_df["Model"].eq("Calibrated XGBoost (OOF)")]

    if not raw.empty and not calibrated.empty:
        raw_row = raw.iloc[0]
        calibrated_row = calibrated.iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw XGBoost ROC-AUC", f"{raw_row['ROC_AUC']:.4f}")
            st.metric("Raw XGBoost PR-AUC", f"{raw_row['PR_AUC']:.4f}")
            st.metric("Raw Brier Score", f"{raw_row['Brier_Score']:.4f}")
            st.metric("Raw Log Loss", f"{raw_row['Log_Loss']:.4f}")
        with col2:
            st.metric("Calibrated ROC-AUC", f"{calibrated_row['ROC_AUC']:.4f}")
            st.metric("Calibrated PR-AUC", f"{calibrated_row['PR_AUC']:.4f}")
            st.metric("Calibrated Brier Score", f"{calibrated_row['Brier_Score']:.4f}")
            st.metric("Calibrated Log Loss", f"{calibrated_row['Log_Loss']:.4f}")

        st.info(
            "Calibration was tested because TeleMetric uses churn probabilities for "
            "risk scoring. In this experiment, calibration improved PR-AUC only slightly "
            "(0.6499 → 0.6506) while ROC-AUC, Brier Score and Log Loss became slightly "
            "worse. Therefore, the evidence does not support presenting calibration as "
            "an overall performance improvement."
        )
else:
    st.warning("model_evaluation.csv was not found, so OOF calibration metrics cannot be displayed.")

st.divider()

# -----------------------------------------------------------------------------
# Final assessment
# -----------------------------------------------------------------------------
st.subheader("5. How to Read the Model Results")
st.markdown(
    """
**What the experiments establish:**

- **Baseline benchmark:** Logistic Regression had the highest mean 5-fold ROC-AUC (0.8473) among the three *untuned* baseline models.
- **Tuned XGBoost:** A separate RandomizedSearchCV experiment found an XGBoost configuration with a CV ROC-AUC of 0.8485, followed by a held-out test ROC-AUC of 0.8415 and PR-AUC of 0.6531.
- **Probability calibration:** Calibration was investigated because TeleMetric uses probabilities for downstream risk scoring, but the current OOF results do not show an overall improvement from calibration.
- **Explainability:** SHAP provides a global view of which features influence the XGBoost predictions.

**Important modeling caveat:** the baseline Logistic Regression and tuned XGBoost scores come from different experiments, so they should not be presented as a definitive head-to-head final comparison. A future model-selection experiment should tune/evaluate all candidate models under the same protocol and compare ROC-AUC, PR-AUC, classification metrics and probability-quality metrics consistently.
"""
)
