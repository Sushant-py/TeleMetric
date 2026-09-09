import streamlit as st
from pathlib import Path

# Automatically find the exact path to the pages folder
PAGES_DIR = Path(__file__).parent / "pages"

# 1. Define the pages using the foolproof paths
home_page = st.Page(str(PAGES_DIR / "home.py"), title="TeleMetric", icon="🚀")
explorer_page = st.Page(str(PAGES_DIR / "01_Customer_Explorer.py"), title="Customer Explorer")
risk_page = st.Page(str(PAGES_DIR / "02_Risk_Intelligence.py"), title="Risk Intelligence")
simulator_page = st.Page(str(PAGES_DIR / "03_Retention_Simulator.py"), title="Retention Simulator")
insights_page = st.Page(str(PAGES_DIR / "04_Model_Insights.py"), title="Model Insights")

# 2. Pass the defined variables into the navigation router
pg = st.navigation([
    home_page, 
    explorer_page, 
    risk_page, 
    simulator_page, 
    insights_page
])

# 3. Run the router
pg.run()