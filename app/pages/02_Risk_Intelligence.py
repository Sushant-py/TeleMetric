import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

data = pd.read_csv(
    BASE_DIR / "outputs" / "customer_risk_scores.csv"
)

st.title("Risk Intelligence")

fig = px.scatter(
    data,
    x="Churn_Probability",
    y="Estimated_Billed_Value",
    color="Risk_Band",
    size="Expected_Revenue_at_Risk",
    hover_data=[
        "customerID",
        "Contract",
        "tenure"
    ]
)

st.plotly_chart(
    fig,
    width="stretch"
)