import streamlit as st
import pandas as pd
from pathlib import Path
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