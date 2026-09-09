import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import streamlit as st

# Define your pages with clean titles
home_page = st.Page("app/pages/home.py", title="TeleMetric", icon="🚀")
explorer_page = st.Page("app/pages/01_Customer_Explorer.py", title="Customer Explorer")
risk_page = st.Page("app/pages/02_Risk_Intelligence.py", title="Risk Intelligence")
simulator_page = st.Page("app/pages/03_Retention_Simulator.py", title="Retention Simulator")
insights_page = st.Page("app/pages/04_Model_Insights.py", title="Model Insights")

# Initialize navigation
pg = st.navigation([home_page, explorer_page, risk_page, simulator_page, insights_page])
pg.run()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TeleMetric",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = (
    OUTPUT_DIR
    / "customer_risk_scores.csv"
)


# ============================================================
# LOAD CUSTOMER RISK DATA
# ============================================================

try:
    data = pd.read_csv(DATA_PATH)

except FileNotFoundError:
    st.error(
        "customer_risk_scores.csv was not found.\n\n"
        f"Expected location:\n{DATA_PATH}"
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

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

st.subheader(
    "Customer Retention & Churn Intelligence"
)

st.write(
    "Explore customer churn patterns, predicted risk, "
    "revenue exposure, and retention priorities."
)

st.write(
    f"Loaded {len(data):,} customers."
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(data)

churn_rate = (
    data["Churn_Target"].mean()
)

monthly_revenue = (
    data["MonthlyCharges"].sum()
)

revenue_at_risk = (
    data["Expected_Revenue_at_Risk"].sum()
)

high_risk_customers = (
    data["Risk_Band"]
    .isin(["High", "Critical"])
    .sum()
)

average_risk = (
    data["Churn_Probability"].mean()
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.1%}"
    )

with col3:
    st.metric(
        "Monthly Revenue",
        f"${monthly_revenue:,.0f}"
    )

with col4:
    st.metric(
        "High-Risk Customers",
        f"{high_risk_customers:,}"
    )


# ============================================================
# SECONDARY KPIs
# ============================================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Expected Revenue at Risk",
        f"${revenue_at_risk:,.0f}"
    )

with col2:
    st.metric(
        "Average Churn Probability",
        f"{average_risk:.1%}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("TeleMetric")

st.sidebar.write(
    "Customer Retention Intelligence"
)

st.sidebar.divider()


# ------------------------------------------------------------
# Risk Filter
# ------------------------------------------------------------

risk_options = sorted(
    data["Risk_Band"]
    .dropna()
    .unique()
)

risk_filter = st.sidebar.multiselect(
    "Risk Band",
    options=risk_options,
    default=risk_options
)


# ------------------------------------------------------------
# Contract Filter
# ------------------------------------------------------------

contract_options = sorted(
    data["Contract"]
    .dropna()
    .unique()
)

contract_filter = st.sidebar.multiselect(
    "Contract",
    options=contract_options,
    default=contract_options
)


# ------------------------------------------------------------
# Internet Service Filter
# ------------------------------------------------------------

internet_options = sorted(
    data["InternetService"]
    .dropna()
    .unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=internet_options,
    default=internet_options
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_data = data[
    data["Risk_Band"].isin(risk_filter)
]

filtered_data = filtered_data[
    filtered_data["Contract"].isin(contract_filter)
]

filtered_data = filtered_data[
    filtered_data["InternetService"].isin(
        internet_filter
    )
]


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

st.divider()

st.subheader("Customer Explorer")

customer_search = st.text_input(
    "Search Customer ID",
    placeholder="Example: 8149-RSOUN"
)


if customer_search:

    customer = data[
        data["customerID"]
        .astype(str)
        .str.contains(
            customer_search,
            case=False,
            na=False
        )
    ]

    if len(customer) == 0:

        st.warning(
            "No matching customer found."
        )

    else:

        st.dataframe(
            customer[
                [
                    "customerID",
                    "Contract",
                    "tenure",
                    "MonthlyCharges",
                    "Churn_Probability",
                    "Risk_Band",
                    "Expected_Revenue_at_Risk",
                    "Retention_Segment"
                ]
            ],
            width="stretch",
            hide_index=True
        )


# ============================================================
# CHURN RATE BY CONTRACT
# ============================================================

st.subheader(
    "Churn Rate by Contract"
)

contract_summary = (
    filtered_data
    .groupby("Contract")
    .agg(
        Churn_Rate=("Churn_Target", "mean"),
        Customers=("customerID", "count")
    )
    .reset_index()
)

contract_summary["Churn_Rate"] *= 100


fig = px.bar(
    contract_summary,
    x="Contract",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate by Contract Type"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_layout(
    yaxis_title="Churn Rate (%)",
    xaxis_title="Contract"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# CUSTOMER RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Customer Risk Distribution"
)

risk_summary = (
    filtered_data["Risk_Band"]
    .value_counts()
    .reset_index()
)

risk_summary.columns = [
    "Risk_Band",
    "Customers"
]


fig = px.bar(
    risk_summary,
    x="Risk_Band",
    y="Customers",
    text="Customers",
    title="Customer Risk Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# RISK VS VALUE
# ============================================================

st.subheader(
    "Customer Risk vs Estimated Value"
)

fig = px.scatter(
    filtered_data,
    x="Churn_Probability",
    y="Estimated_Billed_Value",
    color="Risk_Band",
    size="Expected_Revenue_at_Risk",
    hover_data=[
        "customerID",
        "Contract",
        "MonthlyCharges",
        "tenure",
        "Retention_Segment"
    ],
    title="Risk-Value Matrix"
)

fig.update_xaxes(
    title="Churn Probability"
)

fig.update_yaxes(
    title="Estimated Billed Value"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# TOP RETENTION PRIORITIES
# ============================================================

st.subheader(
    "Top Retention Priorities"
)

top_targets = (
    filtered_data
    .sort_values(
        "Expected_Revenue_at_Risk",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    top_targets[
        [
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "Churn_Probability",
            "Risk_Band",
            "Expected_Revenue_at_Risk",
            "Retention_Segment"
        ]
    ],
    width="stretch",
    hide_index=True
)


