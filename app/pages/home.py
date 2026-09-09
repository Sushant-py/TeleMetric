import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(page_title="TeleMetric", page_icon="📊", layout="wide")

# Custom CSS for Cosmic Background and Gradient Title
cosmic_theme = """
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] {
    background-color: rgba(15, 12, 41, 0.6);
}
.telemetric-title {
    font-size: 5rem;
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

# ============================================================
# PROJECT PATHS & LOAD DATA
# ============================================================
# Adjusted to parent.parent.parent because this file is now in app/pages/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "outputs" / "customer_risk_scores.csv"

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"customer_risk_scores.csv was not found.\n\nExpected location:\n{DATA_PATH}")
    st.stop()

# ============================================================
# TITLE & KPIs
# ============================================================
st.markdown('<p class="telemetric-title">TeleMetric</p>', unsafe_allow_html=True)
st.markdown("### Customer Retention & Churn Intelligence")
st.write("Explore customer churn patterns, predicted risk, revenue exposure, and retention priorities.")
st.write(f"Loaded {len(data):,} customers.")

total_customers = len(data)
churn_rate = data["Churn_Target"].mean()
monthly_revenue = data["MonthlyCharges"].sum()
revenue_at_risk = data["Expected_Revenue_at_Risk"].sum()
high_risk_customers = data["Risk_Band"].isin(["High", "Critical"]).sum()
average_risk = data["Churn_Probability"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Customers", f"{total_customers:,}")
with col2: st.metric("Churn Rate", f"{churn_rate:.1%}")
with col3: st.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")
with col4: st.metric("High-Risk Customers", f"{high_risk_customers:,}")

st.divider()
col1, col2 = st.columns(2)
with col1: st.metric("Expected Revenue at Risk", f"${revenue_at_risk:,.0f}")
with col2: st.metric("Average Churn Probability", f"{average_risk:.1%}")

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("TeleMetric")
st.sidebar.write("Customer Retention Intelligence")
st.sidebar.divider()

risk_options = sorted(data["Risk_Band"].dropna().unique())
risk_filter = st.sidebar.multiselect("Risk Band", options=risk_options, default=risk_options)

contract_options = sorted(data["Contract"].dropna().unique())
contract_filter = st.sidebar.multiselect("Contract", options=contract_options, default=contract_options)

internet_options = sorted(data["InternetService"].dropna().unique())
internet_filter = st.sidebar.multiselect("Internet Service", options=internet_options, default=internet_options)

filtered_data = data[
    data["Risk_Band"].isin(risk_filter) &
    data["Contract"].isin(contract_filter) &
    data["InternetService"].isin(internet_filter)
]

# ============================================================
# CUSTOMER EXPLORER
# ============================================================
st.divider()
st.subheader("Customer Explorer")
customer_search = st.text_input("Search Customer ID", placeholder="Example: 8149-RSOUN")

if customer_search:
    customer = data[data["customerID"].astype(str).str.contains(customer_search, case=False, na=False)]
    if len(customer) == 0:
        st.warning("No matching customer found.")
    else:
        st.dataframe(
            customer[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn_Probability", "Risk_Band", "Expected_Revenue_at_Risk", "Retention_Segment"]],
            use_container_width=True, hide_index=True
        )

# ============================================================
# CHARTS
# ============================================================
st.subheader("Churn Rate by Contract")
contract_summary = filtered_data.groupby("Contract").agg(Churn_Rate=("Churn_Target", "mean"), Customers=("customerID", "count")).reset_index()
contract_summary["Churn_Rate"] *= 100
fig_contract = px.bar(contract_summary, x="Contract", y="Churn_Rate", text="Churn_Rate", title="Churn Rate by Contract Type", color_discrete_sequence=["#3498db"])
fig_contract.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_contract.update_layout(template="plotly_dark", yaxis_title="Churn Rate (%)", xaxis_title="Contract")
st.plotly_chart(fig_contract, use_container_width=True)

st.subheader("Customer Risk Distribution")
risk_summary = filtered_data["Risk_Band"].value_counts().reset_index()
risk_summary.columns = ["Risk_Band", "Customers"]
fig_risk = px.bar(risk_summary, x="Risk_Band", y="Customers", text="Customers", title="Customer Risk Distribution", color="Risk_Band", color_discrete_map={"Low": "#2ecc71", "Medium": "#3498db", "High": "#e74c3c", "Critical": "#900C3F"})
fig_risk.update_layout(template="plotly_dark")
st.plotly_chart(fig_risk, use_container_width=True)

st.subheader("Customer Risk vs Estimated Value")
fig_scatter = px.scatter(
    filtered_data, x="Churn_Probability", y="Estimated_Billed_Value", color="Risk_Band", size="Expected_Revenue_at_Risk",
    hover_data=["customerID", "Contract", "MonthlyCharges", "tenure", "Retention_Segment"],
    title="Risk-Value Matrix",
    color_discrete_map={"Low": "#2ecc71", "Medium": "#3498db", "High": "#e74c3c", "Critical": "#900C3F"}
)
fig_scatter.update_layout(template="plotly_dark")
fig_scatter.update_xaxes(title="Churn Probability")
fig_scatter.update_yaxes(title="Estimated Billed Value")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Top Retention Priorities")
top_targets = filtered_data.sort_values("Expected_Revenue_at_Risk", ascending=False).head(20)
st.dataframe(
    top_targets[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn_Probability", "Risk_Band", "Expected_Revenue_at_Risk", "Retention_Segment"]],
    use_container_width=True, hide_index=True
)