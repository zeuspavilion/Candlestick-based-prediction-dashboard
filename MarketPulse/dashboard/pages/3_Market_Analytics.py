import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.sql.queries import execute_analytical_query
from MarketPulse.database.models import MarketData, Stock

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")
apply_custom_theme()

st.title("📊 Market Analytics")
st.write("Cross-sectional market dashboards, volatility matrices, and return correlations.")

db = SessionLocal()

# 1. Top Performers Cards
st.subheader("🔥 Daily Session Leaders")
col_g, col_l, col_v = st.columns(3)

try:
    gainers = execute_analytical_query(db, "top_gainers", {"limit": 5})
    losers = execute_analytical_query(db, "top_losers", {"limit": 5})
    volumes = execute_analytical_query(db, "highest_volume", {"limit": 5})
    
    with col_g:
        st.write("**Top Gainers**")
        if gainers:
            df_g = pd.DataFrame(gainers)
            st.dataframe(df_g[["ticker", "pct_change"]].rename(columns={"ticker": "Ticker", "pct_change": "Gain %"}), hide_index=True, use_container_width=True)
        else:
            st.info("No data")
            
    with col_l:
        st.write("**Top Losers**")
        if losers:
            df_l = pd.DataFrame(losers)
            st.dataframe(df_l[["ticker", "pct_change"]].rename(columns={"ticker": "Ticker", "pct_change": "Loss %"}), hide_index=True, use_container_width=True)
        else:
            st.info("No data")
            
    with col_v:
        st.write("**Highest Volume**")
        if volumes:
            df_v = pd.DataFrame(volumes)
            st.dataframe(df_v[["ticker", "total_volume"]].rename(columns={"ticker": "Ticker", "total_volume": "Volume (Hourly)"}), hide_index=True, use_container_width=True)
        else:
            st.info("No data")
            
except Exception as e:
    st.error(f"Failed to load market performance: {e}")

st.markdown("---")

# 2. Volatility and Returns Breakdown
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⚡ Volatility by Asset Category")
    try:
        vol_data = execute_analytical_query(db, "rolling_volatility")
        if vol_data:
            df_vol = pd.DataFrame(vol_data)
            fig_vol = px.bar(
                df_vol,
                x="category",
                y="avg_volatility_pct",
                labels={"category": "Sector Category", "avg_volatility_pct": "Average Volatility (%)"},
                color="category",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_vol.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF'),
                showlegend=False
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No volatility indicators parsed yet.")
    except Exception as e:
        st.error(f"Error loading volatility chart: {e}")
        
with col_right:
    st.subheader("📊 Average Daily Returns")
    try:
        ret_data = execute_analytical_query(db, "avg_daily_return")
        if ret_data:
            df_ret = pd.DataFrame(ret_data)
            fig_ret = px.bar(
                df_ret,
                x="ticker",
                y="avg_return_pct",
                labels={"ticker": "Stock Ticker", "avg_return_pct": "Mean Return %"},
                color="avg_return_pct",
                color_continuous_scale="RdYlGn"
            )
            fig_ret.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF')
            )
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.info("No return distributions loaded.")
    except Exception as e:
        st.error(f"Error loading returns chart: {e}")

st.markdown("---")

# 3. Correlation Matrix Heatmap
st.subheader("🔗 Sector Price Correlation Matrix")
st.write("Analyzes price movements across all bank stock counters in the private and public sector category.")

try:
    # Query price history for all stocks to calculate correlation
    stocks = db.query(Stock).all()
    
    if len(stocks) > 1:
        prices_dict = {}
        for s in stocks:
            md_list = db.query(MarketData).filter(MarketData.stock_id == s.id).order_by(MarketData.timestamp.desc()).limit(150).all()
            if md_list:
                # Store closes keyed by timestamp
                closes = {m.timestamp: m.close for m in md_list}
                prices_dict[s.ticker] = pd.Series(closes)
                
        if prices_dict:
            df_prices = pd.DataFrame(prices_dict).sort_index().pct_change().dropna()
            corr_matrix = df_prices.corr()
            
            # Plot correlation heatmap
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale="Viridis",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF'),
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient market data points to construct correlation matrix.")
    else:
        st.info("Database needs more than 1 stock to calculate asset correlations.")
        
except Exception as e:
    st.error(f"Error generating correlation matrix: {e}")

db.close()
