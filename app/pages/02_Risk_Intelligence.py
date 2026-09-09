import streamlit as st
import pandas as pd
import plotly.express as px
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
data = pd.read_csv(BASE_DIR / "outputs" / "customer_risk_scores.csv")

st.title("Risk Intelligence")

# 1. Bubble chart sizing by Expected Revenue at Risk
fig = px.scatter(
    data,
    x="Churn_Probability",
    y="Estimated_Billed_Value",
    color="Risk_Band",
    size="Expected_Revenue_at_Risk",
    color_discrete_map={
        "Low": "#2ecc71",      # Green
        "Medium": "#3498db",   # Blue
        "High": "#e74c3c",     # Red
        "Critical": "#900C3F"  # Dark Red
    },
    hover_data=["customerID", "Contract", "tenure", "MonthlyCharges"],
    title="Risk vs. Estimated Customer Value"
)

fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)

# 2. Executive Analytics Summary Footer
st.markdown("---")
f1, f2, f3 = st.columns(3)

band_counts = data["Risk_Band"].value_counts().to_dict()
f1.markdown(f"**Risk Bands**\nLow: {band_counts.get('Low', 0):,} · Med: {band_counts.get('Medium', 0):,} · High: {band_counts.get('High', 0):,} · Critical: {band_counts.get('Critical', 0):,}")

top_10_sum = data.nlargest(10, "Expected_Revenue_at_Risk")["Expected_Revenue_at_Risk"].sum()
f2.markdown(f"**Top Exposure**\nTop 10 customers represent **${top_10_sum:,.2f}** of expected exposure.")

total_exp = data["Expected_Revenue_at_Risk"].sum()
conc_pct = (top_10_sum / total_exp) * 100 if total_exp > 0 else 0
f3.markdown(f"**Targeting Concentration**\n**{conc_pct:.1f}%** of total exposure is concentrated in the top 10 customers.")