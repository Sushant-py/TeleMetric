"""Generate TeleMetric customer risk scores from the final selected model.

Run from the repository root after running final_model_selection.py:
    python scripts/generate_final_risk_scores.py

The script intentionally uses the saved final_best_model.pkl and the same
feature-engineered customer dataset used by the fair model-selection script.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_customer_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "final_best_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "final_best_model.pkl not found. Run `python scripts/final_model_selection.py` first."
    )

OUTPUT_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = "Churn_Target"
if TARGET not in data.columns:
    data[TARGET] = data["Churn"].map({"Yes": 1, "No": 0})

# The final model-selection pipeline was trained after removing these columns.
X = data.drop(columns=[c for c in ["customerID", "Churn", TARGET] if c in data.columns])

probability = model.predict_proba(X)[:, 1]

data["Raw_Churn_Probability"] = probability
data["Churn_Probability"] = probability

data["Risk_Band"] = pd.cut(
    data["Churn_Probability"],
    bins=[-np.inf, 0.30, 0.60, 0.80, np.inf],
    labels=["Low", "Medium", "High", "Critical"],
    right=False,
).astype(str)

# Customer value and expected revenue exposure.
data["Estimated_Billed_Value"] = data["MonthlyCharges"] * data["tenure"]
data["Expected_Revenue_at_Risk"] = (
    data["Churn_Probability"] * data["Estimated_Billed_Value"]
)

# Retention priority combines churn probability with normalized customer value.
value_min = data["Estimated_Billed_Value"].min()
value_max = data["Estimated_Billed_Value"].max()
if value_max > value_min:
    data["Normalized_Value"] = (
        (data["Estimated_Billed_Value"] - value_min) /
        (value_max - value_min)
    )
else:
    data["Normalized_Value"] = 0.0

data["Retention_Priority_Score"] = (
    data["Churn_Probability"] * data["Normalized_Value"]
)

# Preserve the project's original, interpretable four-segment decision rule:
# high risk = churn probability >= 0.60; high value = billed value >= median.
value_median = data["Estimated_Billed_Value"].median()
data["High_Value"] = data["Estimated_Billed_Value"] >= value_median
data["High_Risk"] = data["Churn_Probability"] >= 0.60

def retention_segment(row):
    if row["High_Value"] and row["High_Risk"]:
        return "Priority Retention"
    if not row["High_Value"] and row["High_Risk"]:
        return "Automated Retention"
    if row["High_Value"] and not row["High_Risk"]:
        return "Loyal / High Value"
    return "Low Priority"


data["Retention_Segment"] = data.apply(retention_segment, axis=1)

risk_columns = [
    "customerID",
    "Churn_Target",
    "Churn_Probability",
    "Raw_Churn_Probability",
    "Risk_Band",
    "Contract",
    "InternetService",
    "tenure",
    "MonthlyCharges",
    "Estimated_Billed_Value",
    "Expected_Revenue_at_Risk",
    "Retention_Priority_Score",
    "Retention_Segment",
]

risk_path = OUTPUT_DIR / "customer_risk_scores.csv"
data[risk_columns].to_csv(risk_path, index=False)

# Refresh segment-level summary from the final model's probabilities.
segment_summary = (
    data.groupby("Retention_Segment")
    .agg(
        Customers=("customerID", "count"),
        Avg_Churn_Risk=("Churn_Probability", "mean"),
        Total_Revenue_Risk=("Expected_Revenue_at_Risk", "sum"),
        Avg_Monthly_Charge=("MonthlyCharges", "mean"),
    )
    .reset_index()
)
segment_summary.to_csv(OUTPUT_DIR / "retention_segment_summary.csv", index=False)

# Refresh top targets using the same priority score used by the simulator.
top_targets = (
    data.sort_values("Retention_Priority_Score", ascending=False)
    [
        [
            "customerID",
            "Churn_Probability",
            "Risk_Band",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "Estimated_Billed_Value",
            "Expected_Revenue_at_Risk",
            "Retention_Priority_Score",
            "Retention_Segment",
        ]
    ]
    .head(100)
)
top_targets.to_csv(OUTPUT_DIR / "top_100_retention_targets.csv", index=False)

# Make the canonical operational model artifact point to the final selected model.
joblib.dump(model, OUTPUT_DIR / "best_xgboost_model.pkl")

print("Final risk-scoring pipeline completed.")
print(f"Customers scored: {len(data):,}")
print(f"Mean churn probability: {data['Churn_Probability'].mean():.4f}")
print(f"Revenue at risk: ${data['Expected_Revenue_at_Risk'].sum():,.2f}")
print(f"Saved: {risk_path}")
print("Updated: retention_segment_summary.csv")
print("Updated: top_100_retention_targets.csv")
print("Updated: best_xgboost_model.pkl")
