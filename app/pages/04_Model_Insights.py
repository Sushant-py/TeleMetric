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

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_RESULTS_PATH = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

SHAP_IMAGE_PATH = (
    OUTPUT_DIR
    / "shap_feature_importance.png"
)

CALIBRATION_IMAGE_PATH = (
    OUTPUT_DIR
    / "xgboost_calibration_curve.png"
)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

try:

    model_results = pd.read_csv(
        MODEL_RESULTS_PATH
    )

except FileNotFoundError:

    st.error(
        "model_comparison.csv was not found "
        "inside the outputs folder."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Model Insights")

st.write(
    "Evaluate model performance, cross-validation stability, "
    "probability calibration, and the factors influencing "
    "customer churn predictions."
)

st.divider()


# ============================================================
# 1. MODEL LEADERBOARD
# ============================================================

st.subheader("Model Leaderboard")

display_results = model_results.copy()


if "ROC_AUC" in display_results.columns:

    display_results["ROC_AUC"] = (
        display_results["ROC_AUC"].round(3)
    )


if "ROC_AUC_STD" in display_results.columns:

    display_results["ROC_AUC_STD"] = (
        display_results["ROC_AUC_STD"].round(3)
    )


if "ROC_AUC" in display_results.columns:

    display_results = (
        display_results
        .sort_values(
            "ROC_AUC",
            ascending=False
        )
    )


st.dataframe(
    display_results,
    width="stretch",
    hide_index=True
)


# ============================================================
# 2. BEST MODEL
# ============================================================

if "ROC_AUC" in model_results.columns:

    best_row = model_results.loc[
        model_results["ROC_AUC"].idxmax()
    ]

    best_model_name = best_row["Model"]

    best_auc = best_row["ROC_AUC"]

else:

    best_model_name = "Unavailable"

    best_auc = 0


st.info(
    f"Highest ROC-AUC in the current model comparison: "
    f"**{best_model_name} ({best_auc:.3f})**"
)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Highest ROC-AUC",
        best_model_name
    )

with col2:

    st.metric(
        "ROC-AUC",
        f"{best_auc:.3f}"
    )


st.caption(
    "The leaderboard ranks the trained models using "
    "cross-validated ROC-AUC."
)


st.divider()


# ============================================================
# 3. ROC-AUC CHART
# ============================================================

if "ROC_AUC" in model_results.columns:

    st.subheader(
        "Cross-Validated ROC-AUC"
    )

    roc_plot = model_results.copy()

    fig = px.bar(
        roc_plot,
        x="Model",
        y="ROC_AUC",
        text="ROC_AUC",
        title="Model ROC-AUC Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="ROC-AUC",
        xaxis_title="",
        yaxis_range=[
            max(
                0,
                roc_plot["ROC_AUC"].min() - 0.05
            ),
            min(
                1,
                roc_plot["ROC_AUC"].max() + 0.05
            )
        ]
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# 4. MODEL STABILITY
# ============================================================

if "ROC_AUC_STD" in model_results.columns:

    st.subheader(
        "Cross-Validation Stability"
    )

    stability_plot = model_results.copy()

    fig = px.bar(
        stability_plot,
        x="Model",
        y="ROC_AUC_STD",
        text="ROC_AUC_STD",
        title="ROC-AUC Variability Across Folds"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="ROC-AUC Standard Deviation",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Lower standard deviation indicates more "
        "consistent performance across folds."
    )


st.divider()


# ============================================================
# 5. NOTEBOOK 7 OOF EVALUATION
# ============================================================

st.header("Notebook 7 | Out-of-Fold Evaluation")

st.write(
    "Notebook 7 evaluates raw XGBoost and calibrated XGBoost "
    "using cross-fitted out-of-fold probability estimates."
)


# ============================================================
# NOTEBOOK 7 RESULTS
# ============================================================

evaluation_df = pd.DataFrame({

    "Metric": [
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Log Loss"
    ],

    "Raw XGBoost": [
        0.8404,
        0.6499,
        0.1378,
        0.4238
    ],

    "Calibrated XGBoost": [
        0.8378,
        0.6506,
        0.1392,
        0.4331
    ]
})


st.subheader(
    "OOF Evaluation Results"
)

st.dataframe(
    evaluation_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# 6. METRIC CARDS
# ============================================================

st.subheader(
    "Predictive Performance"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Raw XGBoost ROC-AUC",
        "0.8404"
    )

with col2:

    st.metric(
        "Calibrated XGBoost ROC-AUC",
        "0.8378"
    )


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Raw XGBoost PR-AUC",
        "0.6499"
    )

with col2:

    st.metric(
        "Calibrated XGBoost PR-AUC",
        "0.6506"
    )


# ============================================================
# 7. PROBABILITY QUALITY
# ============================================================

st.subheader(
    "Probability Reliability"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Raw Brier Score",
        "0.1378"
    )

with col2:

    st.metric(
        "Calibrated Brier Score",
        "0.1392"
    )


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Raw Log Loss",
        "0.4238"
    )

with col2:

    st.metric(
        "Calibrated Log Loss",
        "0.4331"
    )


# ============================================================
# 8. OOF EVALUATION EXPLANATION
# ============================================================

st.subheader(
    "What the OOF Evaluation Means"
)

st.markdown(
    """
**Out-of-fold (OOF) predictions**

The dataset is divided into five folds.

For each fold:

1. The model is trained on the other four folds.
2. Predictions are generated for the held-out fold.
3. The process is repeated until every customer has a
   prediction from a model that did not train on that
   customer's row.

This provides a more honest estimate of model performance
than evaluating predictions produced on the same data used
for training.


**Raw XGBoost**

These are the original XGBoost probability estimates.


**Calibrated XGBoost**

These are probability estimates after applying sigmoid
calibration using cross-fitted validation.
"""
)


# ============================================================
# 9. CALIBRATION CURVE
# ============================================================

st.divider()

st.header(
    "Probability Calibration"
)

st.write(
    "The calibration curve compares predicted churn "
    "probabilities with the observed churn frequency."
)


if CALIBRATION_IMAGE_PATH.exists():

    st.image(
        str(CALIBRATION_IMAGE_PATH),
        caption=(
            "OOF Probability Calibration: "
            "Raw XGBoost vs Calibrated XGBoost"
        ),
        width="stretch"
    )

else:

    st.warning(
        "xgboost_calibration_curve.png was not found."
    )

    st.info(
        "Generate the calibration curve from Notebook 7 "
        "using the calibration cell described below."
    )


# ============================================================
# 10. CALIBRATION INTERPRETATION
# ============================================================

st.subheader(
    "How to Read the Calibration Curve"
)

st.markdown(
    """
The diagonal line represents **perfect calibration**.

For example, a group of customers predicted at approximately
70% churn probability should actually contain about 70%
churners for the predictions to be perfectly calibrated.

A curve closer to the diagonal indicates better probability
calibration.

Calibration is different from ROC-AUC:

- **ROC-AUC** measures ranking/discrimination.
- **PR-AUC** measures precision-recall performance.
- **Brier Score** measures probability accuracy.
- **Log Loss** penalizes incorrect probability estimates.
"""
)


# ============================================================
# 11. CURRENT CALIBRATION RESULT
# ============================================================

st.subheader(
    "Current Calibration Result"
)

st.markdown(
    """
In this experiment:

- Raw XGBoost ROC-AUC: **0.8404**
- Calibrated XGBoost ROC-AUC: **0.8378**
- Raw XGBoost PR-AUC: **0.6499**
- Calibrated XGBoost PR-AUC: **0.6506**
- Raw XGBoost Brier Score: **0.1378**
- Calibrated XGBoost Brier Score: **0.1392**
- Raw XGBoost Log Loss: **0.4238**
- Calibrated XGBoost Log Loss: **0.4331**

Raw XGBoost has the better ROC-AUC, Brier Score,
and Log Loss.

Calibrated XGBoost has a slightly higher PR-AUC.

Therefore, calibration did not improve the overall
probability metrics in this experiment.
"""
)


st.divider()


# ============================================================
# 12. SHAP FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Global Churn Drivers"
)

if SHAP_IMAGE_PATH.exists():

    st.image(
        str(SHAP_IMAGE_PATH),
        caption=(
            "SHAP feature importance showing the variables "
            "influencing churn predictions."
        ),
        width="stretch"
    )

else:

    st.warning(
        "SHAP feature importance image was not found."
    )


# ============================================================
# 13. SHAP INTERPRETATION
# ============================================================

st.subheader(
    "How to Interpret SHAP"
)

st.markdown(
    """
**SHAP importance** shows which variables have the greatest
influence on the model's predictions.

A feature with high SHAP importance has a strong influence
on prediction changes across customers.

SHAP describes model behavior. It should not be interpreted
as proof that a feature causally causes customer churn.
"""
)


# ============================================================
# 14. FINAL MODEL INSIGHTS
# ============================================================

st.divider()

st.subheader(
    "Overall Model Assessment"
)

st.markdown(
    """
TeleMetric evaluates models from two perspectives.

### Predictive performance

The model should correctly distinguish customers who are
likely to churn from customers who are likely to remain.

ROC-AUC and PR-AUC measure this aspect.

### Probability reliability

The model should also produce probabilities that are
meaningful and trustworthy.

Brier Score, Log Loss, and the calibration curve measure
this aspect.

Notebook 7 extends the original model comparison by adding
out-of-fold probability evaluation and calibration analysis.

The current experiment indicates that raw XGBoost provides
the strongest overall probability performance among the
evaluated XGBoost variants.
"""
)