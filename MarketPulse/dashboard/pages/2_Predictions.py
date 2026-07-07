import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from MarketPulse.dashboard.styles import apply_custom_theme, render_kpi_card
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock, Prediction, ModelRun
from MarketPulse.ml.inference import CandlestickPredictor
from MarketPulse.config import MANIFEST_DIR, PROJECT_ROOT

st.set_page_config(page_title="Predictions & XAI", page_icon="🎯", layout="wide")
apply_custom_theme()

st.title("🎯 Prediction Dashboard & Explainable AI")
st.write("Inspect machine learning predictions, probability weights, and attention maps (Grad-CAM).")

db = SessionLocal()

# Load latest predictions
try:
    pred_query = (
        db.query(
            Prediction.id,
            Stock.ticker,
            Prediction.prediction_timestamp,
            Prediction.label_date,
            Prediction.current_close,
            Prediction.next_close,
            Prediction.next_day_return,
            Prediction.predicted_label,
            Prediction.true_label,
            Prediction.confidence,
            Prediction.correct
        )
        .join(Stock, Stock.id == Prediction.stock_id)
        .order_by(Prediction.prediction_timestamp.desc())
        .limit(200)
        .all()
    )
    
    df_preds = pd.DataFrame([{
        "ID": r.id,
        "Ticker": r.ticker,
        "Timestamp": r.prediction_timestamp,
        "Label Date": r.label_date,
        "Current Close": r.current_close,
        "Next Close": r.next_close,
        "Return": f"{r.next_day_return*100.0:.2f}%",
        "Predicted": r.predicted_label,
        "True Label": r.true_label if r.true_label else "Pending",
        "Confidence": f"{r.confidence*100.0:.1f}%",
        "Correct": "Yes" if r.correct else ("No" if r.correct is False else "N/A")
    } for r in pred_query])
except Exception as e:
    st.error(f"Error loading predictions: {e}")
    df_preds = pd.DataFrame()

# Layout
tab_table, tab_xai = st.tabs(["📋 Predictions Register", "🔍 Explainable AI (Grad-CAM)"])

with tab_table:
    st.subheader("Latest Predictions Inferred")
    if not df_preds.empty:
        st.dataframe(df_preds, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions found in the database. Run training and inference or seed records.")

with tab_xai:
    st.subheader("Model Interpretability / Visualizing Attention")
    st.write("Understand what visual features in the candlestick chart (e.g. wicks, body color, volume spikes) the Vision Transformer focuses on.")
    
    # Check if manifests are present
    test_manifest_path = MANIFEST_DIR / "test.csv"
    if test_manifest_path.exists():
        test_df = pd.read_csv(test_manifest_path)
        
        # Filters for selecting sample
        tickers = sorted(list(test_df["ticker"].unique()))
        selected_ticker = st.selectbox("Select Asset Ticker for XAI", tickers)
        
        ticker_samples = test_df[test_df["ticker"] == selected_ticker].reset_index(drop=True)
        
        sample_options = [
            f"Index {idx} | Date: {row['end_date']} | Actual: {row['label']}"
            for idx, row in ticker_samples.iterrows()
        ]
        
        selected_sample_str = st.selectbox("Select Target Sample Close Date", sample_options)
        
        if selected_sample_str:
            sample_idx = int(selected_sample_str.split(" | ")[0].replace("Index ", ""))
            sample_row = ticker_samples.iloc[sample_idx]
            
            # Predict and explain
            st.write("### Neural Network Attention Overlay")
            
            col_img1, col_img2, col_img3 = st.columns(3)
            
            # Instantiating predictor (dynamic loading of checkp)
            predictor = CandlestickPredictor()
            image_relative_path = Path(sample_row["image_path"])
            image_absolute_path = PROJECT_ROOT / image_relative_path
            
            if image_absolute_path.exists():
                try:
                    # Generate explanation
                    explanation = predictor.explain(image_absolute_path)
                    
                    with col_img1:
                        st.write("**Input Candlestick Chart (18-Candle Window)**")
                        st.image(explanation["image"], use_container_width=True)
                        st.caption(f"Asset: {sample_row['ticker']} | Interval Ending: {sample_row['end_date']}")
                        
                    with col_img2:
                        st.write(f"**Attention Heatmap ({explanation['method']})**")
                        # Normalize explanation heatmap for PIL
                        heatmap_img = explanation["heatmap"]
                        st.image(heatmap_img, clamp=True, use_container_width=True)
                        st.caption("Bright regions show high model activation.")
                        
                    with col_img3:
                        st.write("**Attention Overlay**")
                        st.image(explanation["overlay"], use_container_width=True)
                        st.caption(f"Predicted Label: **{explanation['label'].upper()}** | Confidence: **{explanation['confidence']*100.0:.1f}%**")
                        
                except Exception as ex:
                    st.error(f"Failed to generate explanation overlay: {ex}")
            else:
                st.error(f"Image file does not exist: {image_absolute_path}. Run image generation first.")
    else:
        st.info("No test dataset manifests generated yet. Train models and generate manifest files in Settings.")

db.close()
