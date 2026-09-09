import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from PIL import Image
import streamlit as st

# Custom CSS for Cosmic Background and Gradient Title
cosmic_theme = """
<style>
/* Deep Cosmic Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* Sidebar transparency to blend with the cosmic background */
[data-testid="stSidebar"] {
    background-color: rgba(15, 12, 41, 0.6);
}

/* Gradient TeleMetric Title (Red -> Blue -> Green) */
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

# Inject the CSS
st.markdown(cosmic_theme, unsafe_allow_html=True)

# Render the stylized title
st.markdown('<p class="telemetric-title">TeleMetric</p>', unsafe_allow_html=True)

# Continue with the rest of your main page code...
st.markdown("### Customer Retention & Churn Intelligence")
st.set_page_config(page_title="TeleMetric | Model Insights", page_icon="🧠", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_comparison.csv"
SHAP_IMAGE_PATH = OUTPUT_DIR / "shap_feature_importance.png"
CALIBRATION_IMAGE_PATH = OUTPUT_DIR / "xgboost_calibration_curve.png"

try:
    model_results = pd.read_csv(MODEL_RESULTS_PATH)
except FileNotFoundError:
    st.error("model_comparison.csv was not found inside the outputs folder.")
    st.stop()

st.title("Model Comparison & Explainability")
st.write("Model performance and the global SHAP feature-importance artifact produced by the ML pipeline.")

if "ROC_AUC" in model_results.columns:
    best_row = model_results.loc[model_results["ROC_AUC"].idxmax()]
    best_model_name = best_row["Model"]
    best_auc = best_row["ROC_AUC"]
    best_std = best_row["ROC_AUC_STD"] if "ROC_AUC_STD" in best_row else 0
else:
    best_model_name = "Unavailable"
    best_auc = 0
    best_std = 0

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Best model**\n\n### {best_model_name}\nROC-AUC {best_auc*100:.2f}% · CV std {best_std*100:.2f} percentage points")
with col2:
    st.info("**Model selection principle**\n\nCompare ROC-AUC across Logistic Regression, Random Forest and XGBoost. The dashboard reads the project's model comparison output directly.")

st.divider()

st.subheader("Model Leaderboard")
display_results = model_results.copy()
if "ROC_AUC" in display_results.columns:
    display_results["ROC_AUC"] = display_results["ROC_AUC"].round(3)
    display_results = display_results.sort_values("ROC_AUC", ascending=False)
if "ROC_AUC_STD" in display_results.columns:
    display_results["ROC_AUC_STD"] = display_results["ROC_AUC_STD"].round(3)

st.dataframe(display_results, use_container_width=True, hide_index=True)
st.caption("The leaderboard ranks the trained models using cross-validated ROC-AUC.")
st.divider()

if "ROC_AUC" in model_results.columns:
    st.subheader("ROC-AUC Comparison")
    fig_auc = px.bar(
        model_results,
        x="Model",
        y="ROC_AUC",
        text="ROC_AUC",
        color="Model",
        color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"] # Red, Green, Blue
    )
    fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_auc.update_layout(template="plotly_dark", yaxis_title="ROC-AUC (%)", xaxis_title="", yaxis_range=[max(0, model_results["ROC_AUC"].min() - 0.05), min(1, model_results["ROC_AUC"].max() + 0.05)], height=400)
    st.plotly_chart(fig_auc, use_container_width=True)

if "ROC_AUC_STD" in model_results.columns:
    st.subheader("Cross-Validation Stability")
    fig_std = px.bar(
        model_results,
        x="Model",
        y="ROC_AUC_STD",
        text="ROC_AUC_STD",
        color="Model",
        color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"] # Red, Green, Blue
    )
    fig_std.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_std.update_layout(template="plotly_dark", yaxis_title="ROC-AUC Standard Deviation", xaxis_title="", height=400)
    st.plotly_chart(fig_std, use_container_width=True)

st.divider()

st.header("Model Explainability (SHAP)")
st.write("SHAP importance shows which variables have the greatest influence on the model's predictions.")

if SHAP_IMAGE_PATH.exists():
    try:
        img_shap = Image.open(SHAP_IMAGE_PATH)
        st.image(img_shap, caption="Global Feature Importance (TreeSHAP)", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load SHAP image: {e}")
else:
    st.warning("SHAP feature importance image was not found.")

st.info("**SHAP Data**\n\nThe underlying ranked values are available in `outputs/shap_feature_importance.csv`, making the explanation reproducible rather than image-only.")
st.divider()

st.header("Probability Calibration")
if CALIBRATION_IMAGE_PATH.exists():
    try:
        img_calib = Image.open(CALIBRATION_IMAGE_PATH)
        st.image(img_calib, caption="OOF Probability Calibration: Raw XGBoost vs Calibrated XGBoost", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load Calibration image: {e}")
else:
    st.warning("xgboost_calibration_curve.png was not found.")

evaluation_df = pd.DataFrame({
    "Metric": ["ROC-AUC", "PR-AUC", "Brier Score", "Log Loss"],
    "Raw XGBoost": [0.8404, 0.6499, 0.1378, 0.4238],
    "Calibrated XGBoost": [0.8378, 0.6506, 0.1392, 0.4331]
})

st.subheader("OOF Evaluation Results")
st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

st.subheader("Predictive Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Raw XGBoost ROC-AUC", "0.8404")
    st.metric("Raw XGBoost PR-AUC", "0.6499")
with col2:
    st.metric("Calibrated XGBoost ROC-AUC", "0.8378")
    st.metric("Calibrated XGBoost PR-AUC", "0.6506")

st.subheader("Probability Reliability")
col1, col2 = st.columns(2)
with col1:
    st.metric("Raw Brier Score", "0.1378")
    st.metric("Raw Log Loss", "0.4238")
with col2:
    st.metric("Calibrated Brier Score", "0.1392")
    st.metric("Calibrated Log Loss", "0.4331")

st.divider()
st.subheader("Overall Model Assessment")
st.markdown("""
TeleMetric evaluates models from two perspectives:
* **Predictive performance:** The model should correctly distinguish customers who are likely to churn from customers who are likely to remain. ROC-AUC and PR-AUC measure this aspect.
* **Probability reliability:** The model should also produce probabilities that are meaningful and trustworthy. Brier Score, Log Loss, and the calibration curve measure this aspect.

The current experiment indicates that raw XGBoost provides the strongest overall probability performance among the evaluated XGBoost variants.
""")