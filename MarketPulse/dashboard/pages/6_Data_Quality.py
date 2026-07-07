import streamlit as st
import pandas as pd
import plotly.express as px
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import DataQualityReport, PipelineLog

st.set_page_config(page_title="Data Quality & Logs", page_icon="🛡️", layout="wide")
apply_custom_theme()

st.title("🛡️ Data Quality & Pipeline Logs")
st.write("Monitor raw stock ingestion audits, row processing stats, and ETL execution logs.")

db = SessionLocal()

# Load reports
try:
    reports = db.query(DataQualityReport).order_by(DataQualityReport.timestamp.desc()).limit(50).all()
    df_reports = pd.DataFrame([{
        "Report ID": r.id,
        "Timestamp": r.timestamp,
        "Run ID": r.run_id,
        "Table": r.table_name,
        "Processed Rows": r.rows_processed,
        "Skipped Rows": r.rows_skipped,
        "Rejected Rows": r.rows_rejected,
        "Missing Values": r.missing_values,
        "Duplicates": r.duplicate_count,
        "Success Rate": f"{r.success_rate:.2f}%",
        "Failures Details": r.details
    } for r in reports])
except Exception as e:
    st.error(f"Error loading reports: {e}")
    df_reports = pd.DataFrame()

# Load Pipeline Logs
try:
    logs = db.query(PipelineLog).order_by(PipelineLog.timestamp.desc()).limit(150).all()
    df_logs = pd.DataFrame([{
        "Timestamp": r.timestamp,
        "Module": r.module,
        "Level": r.level,
        "Message": r.message,
        "Details": r.details
    } for r in logs])
except Exception as e:
    st.error(f"Error loading logs: {e}")
    df_logs = pd.DataFrame()

# Layout
tab_dq, tab_logs = st.tabs(["🛡️ Data Quality Audits", "📋 System Pipeline Logs"])

with tab_dq:
    st.subheader("Data Validation Run Logs")
    if not df_reports.empty:
        # Show metric summary cards for the latest run
        latest = df_reports.iloc[0]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Latest Processed Rows", latest["Processed Rows"])
        with col_m2:
            st.metric("Skipped/Duplicates", latest["Skipped Rows"])
        with col_m3:
            st.metric("Rejected/Malformed", latest["Rejected Rows"])
        with col_m4:
            st.metric("Ingestion Success Rate", latest["Success Rate"])
            
        st.dataframe(df_reports, use_container_width=True, hide_index=True)
        
        # Plotly pie chart of the latest run breakdown
        st.write("#### Ingestion Records Distribution (Latest Ingestion Run)")
        run_breakdown = pd.DataFrame([
            {"Status": "Success (Loaded)", "Count": int(latest["Processed Rows"]) - int(latest["Skipped Rows"]) - int(latest["Rejected Rows"])},
            {"Status": "Skipped (Duplicate)", "Count": int(latest["Skipped Rows"])},
            {"Status": "Rejected (Errors)", "Count": int(latest["Rejected Rows"])}
        ])
        
        fig_pie = px.pie(
            run_breakdown,
            names="Status",
            values="Count",
            color="Status",
            color_discrete_map={
                "Success (Loaded)": "#00E676",
                "Skipped (Duplicate)": "#FFB74D",
                "Rejected (Errors)": "#FF5252"
            },
            hole=0.3
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No Data Quality audit reports available. Ingest some market data to trigger validation audits.")

with tab_logs:
    st.subheader("Event Pipeline Execution Logs")
    
    # Simple filters
    if not df_logs.empty:
        col_lf1, col_lf2 = st.columns(2)
        with col_lf1:
            levels = ["ALL"] + list(df_logs["Level"].unique())
            selected_level = st.selectbox("Log Severity Level", levels)
        with col_lf2:
            modules = ["ALL"] + list(df_logs["Module"].unique())
            selected_module = st.selectbox("Pipeline Submodule", modules)
            
        # Apply filters
        df_filtered = df_logs.copy()
        if selected_level != "ALL":
            df_filtered = df_filtered[df_filtered["Level"] == selected_level]
        if selected_module != "ALL":
            df_filtered = df_filtered[df_filtered["Module"] == selected_module]
            
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    else:
        st.info("No system execution logs available.")

db.close()
