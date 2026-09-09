import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

try:
    data = pd.read_csv(BASE_DIR / "outputs" / "customer_risk_scores.csv")
except FileNotFoundError:
    st.error("customer_risk_scores.csv not found in the outputs folder.")
    st.stop()

try:
    seg_summary = pd.read_csv(BASE_DIR / "outputs" / "retention_segment_summary.csv")
    strat_comp = pd.read_csv(BASE_DIR / "outputs" / "retention_strategy_comparison.csv")
except FileNotFoundError:
    seg_summary = None
    strat_comp = None

st.title("Retention Simulator")
st.write("Simulate how a limited retention team could prioritize customers based on expected revenue risk.")

capacity = st.slider("Retention team capacity", min_value=100, max_value=2000, value=500, step=100)

risk_sorted = data.sort_values("Expected_Revenue_at_Risk", ascending=False)
selected = risk_sorted.head(capacity)

captured_risk = selected["Expected_Revenue_at_Risk"].sum()
total_risk = data["Expected_Revenue_at_Risk"].sum()
coverage = captured_risk / total_risk if total_risk > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customers Targeted", f"{capacity:,}")
with col2:
    st.metric("Revenue Risk Captured", f"${captured_risk:,.0f}")
with col3:
    st.metric("Risk Coverage", f"{coverage:.1%}")

st.subheader("Priority Customers")
st.dataframe(
    selected[["customerID", "Churn_Probability", "MonthlyCharges", "Expected_Revenue_at_Risk", "Retention_Segment"]],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

if seg_summary is not None and strat_comp is not None and not strat_comp.empty:
    st.markdown("### Strategic Retention Analytics")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Retention Segment Profile")
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(
            x=seg_summary.iloc[:, 0], 
            y=seg_summary["Avg_Churn_Risk"] * 100 if "Avg_Churn_Risk" in seg_summary.columns else seg_summary.iloc[:, 1],
            name="Average Churn Risk %",
            marker_color="#e74c3c" # Red
        ))
        
        rev_col = [c for c in seg_summary.columns if "Rev" in c or "Risk" in c][-1]
        fig_seg.add_trace(go.Bar(
            x=seg_summary.iloc[:, 0],
            y=seg_summary[rev_col],
            name="Revenue Risk",
            marker_color="#3498db" # Blue
        ))
        fig_seg.update_layout(barmode="group", template="plotly_dark", height=380)
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        st.subheader("Retention Strategy Comparison")
        # Dynamically grab columns to avoid KeyError
        x_col = strat_comp.columns[0]
        y_col = strat_comp.columns[1]
        
        fig_strat = px.bar(
            strat_comp,
            x=x_col,
            y=y_col,
            text=strat_comp[y_col].apply(lambda val: f"{val:,.1f}"),
            title="Captured Revenue Exposure by Strategy",
            color=x_col,
            color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db"] # Red, Green, Blue gradient
        )
        fig_strat.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_strat, use_container_width=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Priority Retention", "728 customers", "Avg Risk: 45.9% | Exp: $48,461")
    kpi2.metric("Automated Retention", "1,582 customers", "Avg Risk: 63.8% | Exp: $89,908")
    kpi3.info("**Decision Rule:** Interventions prioritize customers where high probability intersects with significant monthly contract value.")