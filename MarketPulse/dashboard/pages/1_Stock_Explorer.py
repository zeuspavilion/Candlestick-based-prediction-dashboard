import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock, MarketData, TechnicalIndicator

st.set_page_config(page_title="Stock Explorer", page_icon="📈", layout="wide")
apply_custom_theme()

st.title("📈 Stock Price Explorer")
st.write("Analyze historical candlestick pricing data, volume distributions, and moving averages.")

db = SessionLocal()

# Filters
stocks = db.query(Stock).all()
tickers = [s.ticker for s in stocks]

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_ticker = st.selectbox("Select Asset Ticker", tickers if tickers else ["HDFCBANK.NS"])
with col_f2:
    limit = st.slider("Max Hourly Candles to Load", 50, 1000, 300)

if stocks:
    stock = db.query(Stock).filter(Stock.ticker == selected_ticker).first()
    
    # Load Market Data
    md_query = db.query(MarketData).filter(MarketData.stock_id == stock.id).order_by(MarketData.timestamp.desc()).limit(limit).all()
    
    if md_query:
        # Load data into DataFrame
        df_md = pd.DataFrame([{
            "Timestamp": r.timestamp,
            "Open": r.open,
            "High": r.high,
            "Low": r.low,
            "Close": r.close,
            "Volume": r.volume
        } for r in md_query]).sort_values("Timestamp")
        
        # Load indicators (for moving averages)
        ti_query = db.query(TechnicalIndicator).filter(TechnicalIndicator.stock_id == stock.id).order_by(TechnicalIndicator.timestamp.desc()).limit(limit).all()
        df_ti = pd.DataFrame([{
            "Timestamp": r.timestamp,
            "MA_20": r.ma_20,
            "MA_50": r.ma_50
        } for r in ti_query]).sort_values("Timestamp")
        
        # Merge
        df = pd.merge(df_md, df_ti, on="Timestamp", how="left")
        
        # Plotly Subplots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            subplot_titles=('Price & Moving Averages', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df["Timestamp"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candlestick",
                increasing_line_color='#00E676',
                decreasing_line_color='#FF5252'
            ),
            row=1, col=1
        )
        
        # Moving Averages
        if "MA_20" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["Timestamp"], y=df["MA_20"], name="MA 20", line=dict(color="#29B6F6", width=1.5)),
                row=1, col=1
            )
        if "MA_50" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["Timestamp"], y=df["MA_50"], name="MA 50", line=dict(color="#FFB74D", width=1.5)),
                row=1, col=1
            )
            
        # Volume
        fig.add_trace(
            go.Bar(
                x=df["Timestamp"], 
                y=df["Volume"], 
                name="Volume", 
                marker_color='#556270',
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # Style Layout
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(t=50, b=20, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            hovermode="x unified"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#2D333F')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#2D333F')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pricing Summary Card
        st.subheader("📊 Price Action Analytics")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Latest Close", f"INR {df['Close'].iloc[-1]:.2f}", f"{(df['Close'].iloc[-1] - df['Open'].iloc[-1]):.2f} Session delta")
        with col_c2:
            st.metric("Session High", f"INR {df['High'].max():.2f}")
        with col_c3:
            st.metric("Session Low", f"INR {df['Low'].min():.2f}")
        with col_c4:
            st.metric("Avg Hourly Volume", f"{df['Volume'].mean():,.0f} units")
            
    else:
        st.warning("No pricing data found in database. Seed and run the ETL process first.")
else:
    st.info("No Stocks found in the database. Seed the database first.")

db.close()
