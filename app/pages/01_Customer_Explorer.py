import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "customer_risk_scores.csv"
)

data = pd.read_csv(DATA_PATH)

st.title("Customer Explorer")

search = st.text_input(
    "Search Customer ID"
)

if search:

    customer = data[
        data["customerID"]
        .astype(str)
        .str.contains(
            search,
            case=False,
            na=False
        )
    ]

    if customer.empty:

        st.warning(
            "No customer found."
        )

    else:

        row = customer.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Churn Probability",
                f"{row['Churn_Probability']:.1%}"
            )

        with col2:
            st.metric(
                "Monthly Charge",
                f"${row['MonthlyCharges']:.2f}"
            )

        with col3:
            st.metric(
                "Revenue at Risk",
                f"${row['Expected_Revenue_at_Risk']:.2f}"
            )

        st.subheader("Customer Details")

        st.dataframe(
            customer,
            width="stretch",
            hide_index=True
        )