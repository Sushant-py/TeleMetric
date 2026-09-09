import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Resolve base directory for all file loads
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load primary simulator data
try:
    data = pd.read_csv(BASE_DIR / "outputs" / "customer_risk_scores.csv")
except FileNotFoundError:
    st.error("customer_risk_scores.csv not found in the outputs folder.")
    st.stop()

# Load strategic analytics artifacts safely
try:
    seg_summary = pd.read_csv(BASE_DIR / "outputs" / "retention_segment_summary.csv")
    strat_comp = pd.read_csv(BASE_DIR / "outputs" / "retention_strategy_comparison.csv")
except FileNotFoundError:
    seg_summary = None
    strat_comp = None

# ============================================================
# 1. RETENTION SIMULATOR
# ============================================================

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

captured_risk = selected["Expected_Revenue_at_Risk"].sum()
total_risk = data["Expected_Revenue_at_Risk"].sum()
coverage = captured_risk / total_risk if total_risk > 0 else 0

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

st.subheader("Priority Customers")

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
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ============================================================
# 2. STRATEGIC RETENTION ANALYTICS
# ============================================================

if seg_summary is not None and strat_comp is not None:
    st.markdown("### Strategic Retention Analytics")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Retention Segment Profile")
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(
            x=seg_summary["Retention_Segment"],
            y=seg_summary["Avg_Churn_Risk"] * 100,
            name="Average Churn Risk %",
            marker_color="#2b7bba"
        ))
        
        # Verify the exact column name for Revenue Risk
        rev_col = "Revenue_Risk_Thousands" if "Revenue_Risk_Thousands" in seg_summary.columns else "Revenue_Risk"
        if rev_col in seg_summary.columns:
            fig_seg.add_trace(go.Bar(
                x=seg_summary["Retention_Segment"],
                y=seg_summary[rev_col],
                name="Revenue Risk / $1K",
                marker_color="#b85d75"
            ))
            
        fig_seg.update_layout(barmode="group", template="plotly_dark", height=380)
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        st.subheader("Retention Strategy Comparison")
        fig_strat = px.bar(
            strat_comp,
            x="Strategy",
            y="Captured_Exposure_Pct",
            text=strat_comp["Captured_Exposure_Pct"].apply(lambda x: f"{x:.1f}%"),
            title="Captured Revenue Exposure by Strategy",
            color_discrete_sequence=["#1f77b4"]
        )
        fig_strat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_strat, use_container_width=True)

    # Strategic KPI Callouts
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Priority Retention", "728 customers", "Avg Risk: 45.9% | Exp: $48,461")
    kpi2.metric("Automated Retention", "1,582 customers", "Avg Risk: 63.8% | Exp: $89,908")
    kpi3.info("**Decision Rule:** Interventions prioritize customers where high probability intersects with significant monthly contract value.")