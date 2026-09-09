import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TeleMetric | Model Insights",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_comparison.csv"
SHAP_IMAGE_PATH = OUTPUT_DIR / "shap_feature_importance.png"
CALIBRATION_IMAGE_PATH = OUTPUT_DIR / "xgboost_calibration_curve.png"

# ============================================================
# LOAD MODEL COMPARISON
# ============================================================
try:
    model_results = pd.read_csv(MODEL_RESULTS_PATH)
except FileNotFoundError:
    st.error("model_comparison.csv was not found inside the outputs folder.")
    st.stop()

# ============================================================
# HEADER & STRATEGY CARDS (Ported from Java UI)
# ============================================================
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
    st.info(
        f"**Best model**\n\n"
        f"### {best_model_name}\n"
        f"ROC-AUC {best_auc*100:.2f}% · CV std {best_std*100:.2f} percentage points"
    )

with col2:
    st.info(
        "**Model selection principle**\n\n"
        "Compare ROC-AUC across Logistic Regression, Random Forest and XGBoost. "
        "The dashboard reads the project's model comparison output directly."
    )

st.divider()

# ============================================================
# 1. MODEL LEADERBOARD
# ============================================================
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

# ============================================================
# 2. ROC-AUC CHART (Upgraded Visuals)
# ============================================================
if "ROC_AUC" in model_results.columns:
    st.subheader("ROC-AUC Comparison")
    roc_plot = model_results.copy()
    
    fig_auc = px.bar(
        roc_plot,
        x="Model",
        y="ROC_AUC",
        text="ROC_AUC",
        color_discrete_sequence=["#2b7bba"]
    )
    fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_auc.update_layout(
        template="plotly_dark",
        yaxis_title="ROC-AUC (%)",
        xaxis_title="",
        yaxis_range=[
            max(0, roc_plot["ROC_AUC"].min() - 0.05),
            min(1, roc_plot["ROC_AUC"].max() + 0.05)
        ],
        height=400
    )
    st.plotly_chart(fig_auc, use_container_width=True)

# ============================================================
# 3. MODEL STABILITY (Upgraded Visuals)
# ============================================================
if "ROC_AUC_STD" in model_results.columns:
    st.subheader("Cross-Validation Stability")
    stability_plot = model_results.copy()
    
    fig_std = px.bar(
        stability_plot,
        x="Model",
        y="ROC_AUC_STD",
        text="ROC_AUC_STD",
        color_discrete_sequence=["#b85d75"]
    )
    fig_std.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_std.update_layout(
        template="plotly_dark",
        yaxis_title="ROC-AUC Standard Deviation",
        xaxis_title="",
        height=400
    )
    st.plotly_chart(fig_std, use_container_width=True)
    st.caption("Lower standard deviation indicates more consistent performance across folds.")

st.divider()

# ============================================================
# 4. SHAP FEATURE IMPORTANCE (Ported from Java UI)
# ============================================================
st.header("Model Explainability (SHAP)")
st.write("SHAP importance shows which variables have the greatest influence on the model's predictions.")

if SHAP_IMAGE_PATH.exists():
    st.image(str(SHAP_IMAGE_PATH), caption="Global Feature Importance (TreeSHAP)", use_column_width=True)
else:
    st.warning("SHAP feature importance image was not found.")

# Exact text match from the Java UI screenshot
st.info("**SHAP Data**\n\nThe underlying ranked values are available in `outputs/shap_feature_importance.csv`, making the explanation reproducible rather than image-only.")

st.divider()

# ============================================================
# 5. NOTEBOOK 7 OOF EVALUATION
# ============================================================
st.header("Notebook 7 | Out-of-Fold Evaluation")
st.write("Notebook 7 evaluates raw XGBoost and calibrated XGBoost using cross-fitted out-of-fold probability estimates.")

evaluation_df = pd.DataFrame({
    "Metric": ["ROC-AUC", "PR-AUC", "Brier Score", "Log Loss"],
    "Raw XGBoost": [0.8404, 0.6499, 0.1378, 0.4238],
    "Calibrated XGBoost": [0.8378, 0.6506, 0.1392, 0.4331]
})

st.subheader("OOF Evaluation Results")
st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

# METRIC CARDS
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

st.subheader("What the OOF Evaluation Means")
st.markdown("""
**Out-of-fold (OOF) predictions**  
The dataset is divided into five folds. For each fold:
1. The model is trained on the other four folds.
2. Predictions are generated for the held-out fold.
3. The process is repeated until every customer has a prediction from a model that did not train on that customer's row.

This provides a more honest estimate of model performance than evaluating predictions produced on the same data used for training.

**Raw XGBoost**: The original XGBoost probability estimates.  
**Calibrated XGBoost**: Probability estimates after applying sigmoid calibration using cross-fitted validation.
""")

st.divider()

# ============================================================
# 6. CALIBRATION CURVE
# ============================================================
st.header("Probability Calibration")
st.write("The calibration curve compares predicted churn probabilities with the observed churn frequency.")

if CALIBRATION_IMAGE_PATH.exists():
    st.image(
        str(CALIBRATION_IMAGE_PATH),
        caption="OOF Probability Calibration: Raw XGBoost vs Calibrated XGBoost",
        use_column_width=True
    )
else:
    st.warning("xgboost_calibration_curve.png was not found.")
    st.info("Generate the calibration curve from Notebook 7 using the calibration cell described below.")

st.subheader("How to Read the Calibration Curve")
st.markdown("""
The diagonal line represents **perfect calibration**. For example, a group of customers predicted at approximately 70% churn probability should actually contain about 70% churners.

A curve closer to the diagonal indicates better probability calibration.

* **ROC-AUC** measures ranking/discrimination.
* **PR-AUC** measures precision-recall performance.
* **Brier Score** measures probability accuracy.
* **Log Loss** penalizes incorrect probability estimates.
""")

st.subheader("Current Calibration Result")
st.markdown("""
In this experiment, Raw XGBoost has the better ROC-AUC, Brier Score, and Log Loss. Calibrated XGBoost has a slightly higher PR-AUC. Therefore, calibration did not improve the overall probability metrics in this experiment.
""")

st.divider()

# ============================================================
# 7. FINAL MODEL INSIGHTS
# ============================================================
st.subheader("Overall Model Assessment")
st.markdown("""
TeleMetric evaluates models from two perspectives:

### Predictive performance
The model should correctly distinguish customers who are likely to churn from customers who are likely to remain. ROC-AUC and PR-AUC measure this aspect.

### Probability reliability
The model should also produce probabilities that are meaningful and trustworthy. Brier Score, Log Loss, and the calibration curve measure this aspect.

Notebook 7 extends the original model comparison by adding out-of-fold probability evaluation and calibration analysis. The current experiment indicates that raw XGBoost provides the strongest overall probability performance among the evaluated XGBoost variants.
""")