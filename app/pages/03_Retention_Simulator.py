import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

data = pd.read_csv(
    BASE_DIR / "outputs" / "customer_risk_scores.csv"
)

st.title("Retention Simulator")

st.write(
    "Simulate how a limited retention team could "
    "prioritize customers based on expected revenue risk."
)

capacity = st.slider(
    "Retention team capacity",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

risk_sorted = data.sort_values(
    "Expected_Revenue_at_Risk",
    ascending=False
)

selected = risk_sorted.head(capacity)

captured_risk = (
    selected["Expected_Revenue_at_Risk"]
    .sum()
)

total_risk = (
    data["Expected_Revenue_at_Risk"]
    .sum()
)

coverage = (
    captured_risk / total_risk
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Customers Targeted",
        f"{capacity:,}"
    )

with col2:
    st.metric(
        "Revenue Risk Captured",
        f"${captured_risk:,.0f}"
    )

with col3:
    st.metric(
        "Risk Coverage",
        f"{coverage:.1%}"
    )

st.subheader(
    "Priority Customers"
)

st.dataframe(
    selected[
        [
            "customerID",
            "Churn_Probability",
            "MonthlyCharges",
            "Expected_Revenue_at_Risk",
            "Retention_Segment"
        ]
    ],
    width="stretch",
    hide_index=True
)