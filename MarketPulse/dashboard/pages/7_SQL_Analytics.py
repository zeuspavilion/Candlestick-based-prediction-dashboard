import streamlit as st
import pandas as pd
from sqlalchemy import text
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.sql.queries import ANALYTICAL_QUERIES

st.set_page_config(page_title="SQL Analytics", page_icon="🛢️", layout="wide")
apply_custom_theme()

st.title("🛢️ Relational Database SQL Analytics")
st.write("Write, inspect, and execute raw SQL queries against the active relational database.")

db = SessionLocal()

# 1. Predefined Analytical Queries
st.subheader("📚 Predefined Analytical Library")
st.write("Execute pre-compiled SQL queries that perform complex financial ratios, accuracy calculations, or market anomalies detection.")

query_options = list(ANALYTICAL_QUERIES.keys())
selected_query_name = st.selectbox("Select SQL Query Template", query_options)

if selected_query_name:
    raw_sql = ANALYTICAL_QUERIES[selected_query_name]
    
    st.markdown("**SQL Statement**:")
    st.code(raw_sql, language="sql")
    
    # Optional parameters
    params = {}
    if ":limit" in raw_sql:
        limit_val = st.number_input("Parameter: Limit rows", min_value=1, max_value=100, value=5, key="preset_limit")
        params["limit"] = limit_val
        
    if st.button("Run Preset Query", key="run_preset"):
        with st.spinner("Executing SQL query against relational engine..."):
            try:
                result = db.execute(text(raw_sql), params)
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    if not df.empty:
                        st.success(f"Query returned {len(df)} records.")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Query executed successfully, but returned 0 rows.")
                else:
                    db.commit()
                    st.success("Query executed successfully (no rows returned).")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

st.markdown("---")

# 2. Custom SQL Query Editor
st.subheader("💻 Custom SQL Query Console")
st.write("Write your own SQL queries against tables: `stocks`, `market_data`, `technical_indicators`, `predictions`, `model_runs`, `pipeline_logs`, `data_quality_reports`, `watchlist`.")

custom_sql = st.text_area(
    "SQL Input Console", 
    value="SELECT * FROM stocks LIMIT 5;",
    height=180
)

# Custom limit parameters or direct execution
if st.button("Execute Console Query", key="run_custom"):
    with st.spinner("Processing raw SQL statement..."):
        try:
            result = db.execute(text(custom_sql))
            if result.returns_rows:
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                if not df.empty:
                    st.success(f"Custom query returned {len(df)} records.")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Query executed successfully, but returned 0 rows.")
            else:
                db.commit()
                st.success("Query executed successfully (DML command applied).")
        except Exception as e:
            st.error(f"SQL Syntax / Database Exception: {e}")

db.close()
st.sidebar.caption("System supports SQL-92 query standards for relational engines.")
