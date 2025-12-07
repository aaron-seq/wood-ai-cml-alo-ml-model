"""Streamlit Dashboard for CML Optimization - Wood Engineering Customization.

This module provides an interactive web dashboard for the CML Optimization system,
customized for Wood Engineering with specific branding, educational resources,
and strict data handling policies.
"""

from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use proper imports
try:
    from app.forecasting import CMLForecaster
    from app.sme_override import SMEOverrideManager
    from app.utils import validate_cml_dataframe
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("Failed to import required modules. Please check your installation.")
    st.stop()

# Constants
DEFAULT_MINIMUM_THICKNESS = 3.0
DEFAULT_SAFETY_FACTOR = 1.5
MAX_PREVIEW_ROWS = 10
MAX_RESULTS_DISPLAY = 100
LOGO_PATH = Path("Wood-logo-WHITE-45mm.png")

# Page configuration
st.set_page_config(
    page_title="Wood Engineering - CML Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Wood Engineering Theme consistency
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #1E1E1E;
    }
    h1, h2, h3 {
        color: #00AEEF !important;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #00AEEF;
        color: white;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Branding
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
else:
    st.sidebar.title("WOOD ENGINEERING")

st.sidebar.header("CML Analysis Platform")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Upload & Analyze", "Forecasting", "SME Overrides", "Reports", "How It Works"],
)

# --- Session State Management ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

# --- Helpers ---
@st.cache_resource
def get_forecaster() -> CMLForecaster:
    return CMLForecaster()

@st.cache_resource
def get_sme_manager() -> SMEOverrideManager:
    return SMEOverrideManager()

def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

try:
    forecaster = get_forecaster()
    sme_manager = get_sme_manager()
except Exception:
    st.stop()

# --- Pages ---

if page == "Overview":
    st.title("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Status", "Operational")
    with col2:
        st.metric("Model Version", "RF-Ensemble v2.1")
    with col3:
        if st.session_state['data'] is not None:
            st.metric("Active Dataset", f"{len(st.session_state['data'])} Records")
        else:
            st.metric("Active Dataset", "None")

    st.markdown("---")

    if st.session_state['data'] is None:
        st.info("👋 Welcome to the Wood Engineering CML Analysis Tool.")
        st.warning("⚠️ No data loaded. Please go to the **Upload & Analyze** page to begin.")
    else:
        df = st.session_state['data']
        st.subheader("Current Dataset Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            # Commodity distribution
            commodity_counts = df["commodity"].value_counts()
            fig = px.pie(
                values=commodity_counts.values,
                names=commodity_counts.index,
                title="Commodity Distribution",
                hole=0.4, 
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Feature type distribution
            feature_counts = df["feature_type"].value_counts()
            fig = px.bar(
                x=feature_counts.index,
                y=feature_counts.values,
                title="Feature Types",
                color_discrete_sequence=['#00AEEF']
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Upload & Analyze":
    st.title("Data Upload & Analysis")
    
    st.markdown("""
    Upload your Condition Monitoring Location (CML) data for automated analysis.
    The system uses our proprietary ML model to recommend optimal elimination candidates.
    """)

    uploaded_file = st.file_uploader("Select CML Data File", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            # Validate
            validation = validate_cml_dataframe(df)
            if validation["valid"]:
                st.session_state['data'] = df
                st.success(f"✅ Loaded {len(df)} records successfully.")
                
                if st.button("Run ML Analysis", type="primary"):
                    with st.spinner("Analyzing CML patterns..."):
                        try:
                            from api_client import score_cml_data, check_api_health
                            
                            if not check_api_health():
                                st.error("⚠️ API server is not responding.")
                            else:
                                # Reset file pointer
                                uploaded_file.seek(0)
                                result = score_cml_data(uploaded_file)
                                
                                if result:
                                    st.session_state['analysis_results'] = result
                                    st.success("Analysis complete.")
                        except ImportError:
                           st.warning("API Client not found. Ensure backend is running.")
                        except Exception as e:
                            st.error(f"Error during scoring: {e}")

            else:
                st.error("Validation failed.")
                for err in validation["errors"]:
                    st.error(f"- {err}")

    # Display Results if available
    if st.session_state['analysis_results']:
        results = st.session_state['analysis_results']['results']
        res_df = pd.DataFrame(results)
        
        st.subheader("Optimization Recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            elim_count = len(res_df[res_df['predicted_elimination_flag'] == 1])
            st.metric("Candidates for Elimination", elim_count, delta="Optimization Opportunity")
        with col2:
            keep_count = len(res_df[res_df['predicted_elimination_flag'] == 0])
            st.metric("Critical Monitoring Points", keep_count)
            
        st.dataframe(res_df.style.apply(lambda x: ['background-color: #3d0000' if x.predicted_elimination_flag == 1 else '' for i in x], axis=1))
        
        # Download button
        csv = res_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Report",
            data=csv,
            file_name="wood_cml_optimization_results.csv",
            mime="text/csv",
        )


elif page == "How It Works":
    st.title("Machine Learning Model Explained")
    
    st.markdown("""
    ### 🧠 The Intelligence Behind the Decision
    
    This application utilizes an advanced **Random Forest Ensemble** model to evaluate reliability. Unlike simple threshold-based rules, this model considers non-linear interactions between multiple variables to mimic the decision-making process of a senior corrosion engineer.

    #### 1. The Core Algorithm: Random Forest
    Imagine consulting 100 different reliability experts. Each expert (Decision Tree) looks at the data and votes on whether a CML should be kept or eliminated.
    *   **Diversity**: Each tree looks at a slightly different subset of data and features.
    *   **Voting**: The final decision is based on the majority vote of these 100 trees.
    *   **Robustness**: This prevents a single outlier data point from skewing the result.

    #### 2. Key Decision Indicators (Feature Importance)
    The model creates a weighted analysis based on the following factors, ranked by influence:

    1.  **Corrosion/Thickness Ratio (35%)**: The most critical indicator. It measures how fast the asset is degrading relative to its remaining wall thickness. A high ratio indicates urgent risk.
    2.  **Average Corrosion Rate (25%)**: The raw speed of degradation.
    3.  **Risk Score (15%)**: A composite metric derived from industry standards (API 570).
    4.  **Remaining Life (10%)**: Years until minimum thickness is breached.
    5.  **Inspection History (5%)**: Timespan since last active verification.
    
    #### 3. Understanding the Confidence Score
    The "Confidence" is not a guess; it is a calculated statistical probability.
    
    *   **Calculation**: If 95 out of 100 trees vote to "ELIMINATE", the probability is 0.95.
    *   **Usage**: 
        *   **High Confidence (>0.8)**: Strong consensus. clear patterns matching historical safe-to-eliminate cases.
        *   **Moderate Confidence (0.6 - 0.8)**: Most indicators align, but some edge cases exist. **Engineer Review Recommended.**
        *   **Low Confidence (<0.6)**: The model detects conflicting signals (e.g., low corrosion rate but very thin wall). **Manual Review Required.**

    #### 4. Human-in-the-Loop
    This tool is a **Decision Support System**, not a replacement. It filters the noise so you can focus your expertise on the complex cases (Moderate/Low confidence).
    """)
    
    # Visualizing Feature Importance (Conceptual)
    importance_data = pd.DataFrame({
        'Feature': ['Corrosion/Thick Ratio', 'Corrosion Rate', 'Risk Score', 'Remaining Life', 'Inspection Age', 'Commodity Type'],
        'Weight': [35, 25, 15, 10, 10, 5]
    })
    fig = px.bar(importance_data, x='Weight', y='Feature', orientation='h', title='Model Decision Weightage', color='Weight', color_continuous_scale='Blues')
    st.plotly_chart(fig)

elif page == "Forecasting":
    st.title("Lifecycle Forecasting")
    if st.session_state['data'] is None:
        st.warning("Please upload data in 'Upload & Analyze' first.")
    else:
        df = st.session_state['data']
        
        col1, col2 = st.columns(2)
        with col1:
            min_thickness = st.number_input("Minimum Required Thickness (mm)", 1.0, 10.0, DEFAULT_MINIMUM_THICKNESS, 0.5)
        with col2:
            safety_factor = st.number_input("Safety Factor", 1.0, 3.0, DEFAULT_SAFETY_FACTOR, 0.1)

        if st.button("Generate Forecasts", type="primary"):
             try:
                custom_forecaster = CMLForecaster(minimum_thickness=min_thickness, safety_factor=safety_factor)
                forecast_df = custom_forecaster.forecast_batch(df)
                
                # Summary Stats
                summary = custom_forecaster.generate_forecast_summary(df)
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Remaining Life", f"{summary['avg_remaining_life_years']:.1f} yrs")
                c2.metric("Critical CMLs", summary['critical_cmls'])
                c3.metric("High Risk CMLs", summary['high_risk_cmls'])
                
                # Detailed Table
                st.subheader("Forecast Details")
                st.dataframe(forecast_df)
                
             except Exception as e:
                st.error(f"Forecasting error: {e}")

elif page == "SME Overrides":
    st.title("Expert Override Management")
    st.markdown("Record and track manual engineering decisions that deviate from model recommendations.")
    
    with st.expander("Add New Override", expanded=False):
        with st.form("override_form"):
            col1, col2 = st.columns(2)
            with col1:
                cml_id = st.text_input("CML ID", placeholder="CML-001")
                decision = st.selectbox("Decision", ["KEEP", "ELIMINATE"])
            with col2:
                sme_name = st.text_input("SME Name", placeholder="Dr. John Smith")

            reason = st.text_area("Reason for Override", height=100)
            submitted = st.form_submit_button("Submit Override")
            
            if submitted and cml_id and reason:
                sme_manager.add_override(cml_id, decision, reason, sme_name)
                st.success("Override recorded.")
    
    # Show overrides
    overrides = sme_manager.get_all_overrides()
    if overrides:
        st.dataframe(pd.DataFrame(overrides))
    else:
        st.info("No overrides recorded.")

elif page == "Reports":
    st.title("Comprehensive Reports")
    if st.session_state['data'] is None:
         st.warning("No data available for reporting.")
    else:
        df = st.session_state['data']
        st.metric("Total Assets Analyzed", len(df))
        
        # Interactive Scatter Plot
        fig = px.scatter(
            df, 
            x="average_corrosion_rate", 
            y="thickness_mm", 
            color="commodity",
            size="risk_score" if "risk_score" in df.columns else None,
            title="Corrosion Rate vs Thickness by Commodity",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #888;'>
    <small>Powered by</small><br>
    <b>Wood Engineering AI Solutions</b><br>
    <small>© 2025 Wood PLC</small>
</div>
""", unsafe_allow_html=True)
