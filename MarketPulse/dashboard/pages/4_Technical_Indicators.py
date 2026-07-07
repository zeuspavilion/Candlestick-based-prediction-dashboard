import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock, MarketData, TechnicalIndicator

st.set_page_config(page_title="Technical Indicators", page_icon="📊", layout="wide")
apply_custom_theme()

st.title("📊 Technical Indicators & Patterns")
st.write("Examine RSI, MACD, and detect rule-based candlestick chart patterns (Doji, Hammer, Engulfing).")

db = SessionLocal()

# Filters
stocks = db.query(Stock).all()
tickers = [s.ticker for s in stocks]

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_ticker = st.selectbox("Select Stock", tickers if tickers else ["HDFCBANK.NS"])
with col_f2:
    limit = st.slider("Hourly Candles", 50, 500, 150)

if stocks:
    stock = db.query(Stock).filter(Stock.ticker == selected_ticker).first()
    
    # Load Market Data & Indicators
    md_query = db.query(MarketData).filter(MarketData.stock_id == stock.id).order_by(MarketData.timestamp.desc()).limit(limit).all()
    ti_query = db.query(TechnicalIndicator).filter(TechnicalIndicator.stock_id == stock.id).order_by(TechnicalIndicator.timestamp.desc()).limit(limit).all()
    
    if md_query and ti_query:
        df_md = pd.DataFrame([{
            "Timestamp": r.timestamp,
            "Open": r.open,
            "High": r.high,
            "Low": r.low,
            "Close": r.close
        } for r in md_query]).sort_values("Timestamp")
        
        df_ti = pd.DataFrame([{
            "Timestamp": r.timestamp,
            "RSI": r.rsi,
            "MACD": r.macd,
            "MACD_Signal": r.macd_signal,
            "MACD_Hist": r.macd_hist,
            "Doji": r.doji,
            "Hammer": r.hammer,
            "Bullish_Engulfing": r.bullish_engulfing,
            "Bearish_Engulfing": r.bearish_engulfing
        } for r in ti_query]).sort_values("Timestamp")
        
        # Merge
        df = pd.merge(df_md, df_ti, on="Timestamp", how="inner")
        
        # Create Subplots: Row 1 = Price + Patterns, Row 2 = RSI, Row 3 = MACD
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Patterns Overlay', 'RSI (14)', 'MACD (12, 26, 9)'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df["Timestamp"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Price", increasing_line_color='#00E676', decreasing_line_color='#FF5252'
            ),
            row=1, col=1
        )
        
        # Overlay Doji markers
        doji_df = df[df["Doji"]]
        if not doji_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=doji_df["Timestamp"], y=doji_df["High"] + (doji_df["High"] * 0.002),
                    mode="markers", marker=dict(symbol="circle", size=8, color="#FFFF00"),
                    name="Doji Signal"
                ),
                row=1, col=1
            )
            
        # Overlay Hammer markers
        hammer_df = df[df["Hammer"]]
        if not hammer_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hammer_df["Timestamp"], y=hammer_df["Low"] - (hammer_df["Low"] * 0.002),
                    mode="markers", marker=dict(symbol="triangle-up", size=9, color="#00E5FF"),
                    name="Hammer Signal"
                ),
                row=1, col=1
            )
            
        # RSI Subplot
        fig.add_trace(
            go.Scatter(x=df["Timestamp"], y=df["RSI"], name="RSI", line=dict(color="#E040FB", width=1.5)),
            row=2, col=1
        )
        # RSI horizontal reference bands
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", line_width=1, row=2, col=1)
        
        # MACD Subplot
        fig.add_trace(
            go.Scatter(x=df["Timestamp"], y=df["MACD"], name="MACD", line=dict(color="#29B6F6", width=1.2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["Timestamp"], y=df["MACD_Signal"], name="Signal", line=dict(color="#FFB74D", width=1.2)),
            row=3, col=1
        )
        # MACD Hist bars
        fig.add_trace(
            go.Bar(x=df["Timestamp"], y=df["MACD_Hist"], name="Histogram", marker_color="#78909C", opacity=0.7),
            row=3, col=1
        )
        
        # Style Layout
        fig.update_layout(
            height=700,
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
        
        # Pattern signals summary
        st.subheader("📋 Detected Candlestick Signals (Chronological)")
        patterns_list = []
        for idx, row in df.iterrows():
            triggers = []
            if row["Doji"]: triggers.append("Doji")
            if row["Hammer"]: triggers.append("Hammer")
            if row["Bullish_Engulfing"]: triggers.append("Bullish Engulfing")
            if row["Bearish_Engulfing"]: triggers.append("Bearish Engulfing")
            
            if triggers:
                patterns_list.append({
                    "Timestamp": row["Timestamp"],
                    "Close Price (INR)": f"{row['Close']:.2f}",
                    "Signals Triggered": ", ".join(triggers)
                })
                
        if patterns_list:
            df_pat = pd.DataFrame(patterns_list).sort_values("Timestamp", ascending=False)
            st.dataframe(df_pat, use_container_width=True, hide_index=True)
        else:
            st.info("No classical candlestick patterns detected in the current historical window.")
            
    else:
        st.warning("Technical indicators or price data is missing. Please ingest data first.")
else:
    st.info("No Stocks found in the database. Seed the database first.")

db.close()
