import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="TeleMetric | Model Insights", page_icon="🧠", layout="wide")

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

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FINAL_RESULTS_PATH = OUTPUT_DIR / "final_model_selection.csv"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "final_model_selection.json"
SHAP_IMAGE_PATH = OUTPUT_DIR / "shap_feature_importance.png"
CALIBRATION_IMAGE_PATH = OUTPUT_DIR / "xgboost_calibration_curve.png"
EVALUATION_PATH = OUTPUT_DIR / "model_evaluation.csv"

try:
    final_results = pd.read_csv(FINAL_RESULTS_PATH)
except FileNotFoundError:
    st.error("Final model-selection results are missing. Run scripts/final_model_selection.py first.")
    st.stop()

try:
    with open(FINAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        final_summary = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    final_summary = {}

st.title("Model Comparison & Explainability")
st.write(
    "Final model selection, held-out performance, probability calibration, "
    "and global SHAP explainability."
)

final_results = final_results.sort_values("CV_ROC_AUC", ascending=False).reset_index(drop=True)
winner = final_results.iloc[0]
winner_name = final_summary.get("selected_model", winner["Model"])

st.success(
    f"**Final model selected: {winner_name}** — it achieved the strongest "
    "cross-validated ROC-AUC and the strongest held-out PR-AUC among the tuned candidates."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Final Model", winner_name)
with col2:
    st.metric("CV ROC-AUC", f"{winner['CV_ROC_AUC']:.4f}")
with col3:
    st.metric("Test PR-AUC", f"{winner['Test_PR_AUC']:.4f}")

st.divider()

st.subheader("1. Final Model Comparison")
st.write(
    "Logistic Regression, Random Forest and XGBoost were tuned and evaluated "
    "under the same 80/20 stratified train-test split, 5-fold StratifiedKFold "
    "cross-validation, and ROC-AUC selection objective. The test set remained "
    "untouched during model selection."
)

comparison = final_results[
    [
        "Model",
        "CV_ROC_AUC",
        "CV_ROC_AUC_STD",
        "Test_ROC_AUC",
        "Test_PR_AUC",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
    ]
].copy()

percentage_cols = [
    "CV_ROC_AUC",
    "CV_ROC_AUC_STD",
    "Test_ROC_AUC",
    "Test_PR_AUC",
    "Test_Accuracy",
    "Test_Precision",
    "Test_Recall",
    "Test_F1",
]

st.dataframe(
    comparison.style.format({c: "{:.3f}" for c in percentage_cols}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Higher ROC-AUC and PR-AUC indicate stronger ranking performance. "
    "The held-out test metrics provide the final generalization check."
)

fig = px.bar(
    final_results,
    x="Model",
    y="CV_ROC_AUC",
    text="CV_ROC_AUC",
    color="Model",
    color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"],
    title="Tuned Model Comparison — 5-Fold CV ROC-AUC",
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(template="plotly_dark", yaxis_title="CV ROC-AUC", xaxis_title="", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"**Why {winner_name}?** It ranked first on the primary cross-validation "
    f"selection metric with a CV ROC-AUC of **{winner['CV_ROC_AUC']:.4f}**. "
    f"On the untouched test set it achieved a ROC-AUC of **{winner['Test_ROC_AUC']:.4f}** "
    f"and PR-AUC of **{winner['Test_PR_AUC']:.4f}**."
)

st.divider()

st.subheader("2. Selected XGBoost Configuration")
if winner_name == "XGBoost":
    try:
        params = json.loads(winner["Best_Params"])
        params_df = pd.DataFrame(
            [{"Hyperparameter": key.replace("model__", ""), "Selected value": value} for key, value in params.items()]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    except (TypeError, json.JSONDecodeError):
        st.write("Selected hyperparameters are stored in final_model_selection.csv.")
else:
    st.write("The selected model is not XGBoost; its tuned configuration is stored in final_model_selection.csv.")

st.divider()

st.header("3. Model Explainability (SHAP)")
st.write(
    "SHAP (SHapley Additive exPlanations) estimates how much each feature contributes "
    "to model predictions. This global importance view explains the tree-based XGBoost "
    "model used in the TeleMetric modeling workflow."
)

if SHAP_IMAGE_PATH.exists():
    img_shap = Image.open(SHAP_IMAGE_PATH)
    st.image(img_shap, caption="Global Feature Importance (TreeSHAP)", use_container_width=True)
    st.caption("Global feature importance across the customer population.")

st.divider()

st.header("4. Probability Calibration")
st.write(
    "Calibration checks whether predicted churn probabilities correspond well to "
    "observed churn frequencies. It is evaluated separately from ranking performance."
)

if CALIBRATION_IMAGE_PATH.exists():
    img_calib = Image.open(CALIBRATION_IMAGE_PATH)
    st.image(
        img_calib,
        caption="OOF Probability Calibration: Raw XGBoost vs Calibrated XGBoost",
        use_container_width=True,
    )

if EVALUATION_PATH.exists():
    evaluation_df = pd.read_csv(EVALUATION_PATH)
    st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

    raw = evaluation_df[evaluation_df["Model"].astype(str).str.contains("raw", case=False, na=False)]
    calibrated = evaluation_df[evaluation_df["Model"].astype(str).str.contains("Calibrated", case=False, na=False)]

    if not raw.empty and not calibrated.empty:
        raw_row = raw.iloc[0]
        calibrated_row = calibrated.iloc[0]
        st.write(
            "The raw XGBoost probabilities remain the operational choice for TeleMetric: "
            "calibration produced only a very small PR-AUC increase while ROC-AUC, Brier "
            "Score and Log Loss were worse in the available OOF evaluation."
        )

st.divider()

st.subheader("5. Final Takeaway")
st.markdown(
    f"""
### {winner_name} is the final model

TeleMetric uses a consistent model-selection protocol rather than comparing results from different experiments. All three candidate models were tuned under the same cross-validation setup and evaluated on the same held-out test set.

**Final result:** {winner_name} achieved **{winner['CV_ROC_AUC']:.2%} CV ROC-AUC**, **{winner['Test_ROC_AUC']:.2%} test ROC-AUC**, and **{winner['Test_PR_AUC']:.2%} test PR-AUC**.

The selected model supplies customer-level churn probabilities, which are then combined with customer value to estimate expected revenue at risk. TeleMetric uses that risk-value combination to prioritize retention effort.
"""
)
