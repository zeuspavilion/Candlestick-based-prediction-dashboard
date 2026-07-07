import streamlit as st
import datetime
import requests
from MarketPulse.dashboard.styles import apply_custom_theme
from MarketPulse.database.connection import SessionLocal
from MarketPulse.database.models import Stock, MarketData, Prediction, ModelRun, DataQualityReport, PipelineLog
from MarketPulse.etl.pipeline import ETLPipeline
from MarketPulse.ml.training import ModelTrainer
from MarketPulse.config import DATABASE_URL, TICKERS_FLAT

st.set_page_config(page_title="Settings & Controls", page_icon="🔧", layout="wide")
apply_custom_theme()

st.title("🔧 Platform Settings & Operations")
st.write("Control databases, trigger data ingestion schedules, and launch deep learning model training jobs.")

db = SessionLocal()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔌 Connection & Database Status")
    st.write(f"**Database Connection String**:")
    st.code(DATABASE_URL)
    
    # Quick health check stats
    try:
        stocks_count = db.query(Stock).count()
        data_count = db.query(MarketData).count()
        pred_count = db.query(Prediction).count()
        run_count = db.query(ModelRun).count()
        
        st.write(f"- **Stocks Registered**: `{stocks_count}`")
        st.write(f"- **Price Records (hourly)**: `{data_count}`")
        st.write(f"- **Inference Predictions**: `{pred_count}`")
        st.write(f"- **Trained Model Run Logs**: `{run_count}`")
    except Exception as e:
        st.error(f"Failed to load status details: {e}")
        
    st.markdown("---")
    
    st.subheader("🔄 Ingest Market Data (ETL)")
    st.write("Trigger historical downloads from Yahoo Finance, clean fields, calculate technical indicators, and load results.")
    
    force_dl = st.checkbox("Force download (overrides local CSV cache)")
    
    if st.button("Trigger Full Ingestion In thread"):
        with st.spinner("Downloading and processing stocks..."):
            pipeline = ETLPipeline()
            try:
                pipeline.run(force_download=force_dl)
                st.success("ETL Pipeline completed successfully! Refresh home page to view data.")
            except Exception as e:
                st.error(f"ETL pipeline failed: {e}")
            finally:
                pipeline.close()

with col_right:
    st.subheader("🧠 Deep Learning Training")
    st.write("Train Vision Transformer (ViT), ResNet18, or Custom CNN models on generated candlestick images.")
    
    model_name = st.selectbox("Model Architecture", ["custom_cnn", "resnet18", "vit_b_16"])
    epochs = st.slider("Training Epochs", 1, 10, 2)
    
    if st.button("Launch Model Training"):
        with st.spinner(f"Training {model_name} for {epochs} epochs in this session thread..."):
            try:
                # First check if manifests are present
                from MarketPulse.config import MANIFEST_DIR
                if not (MANIFEST_DIR / "train.csv").exists():
                    st.info("Generating dataset splits & manifests first from raw stock database records...")
                    # Generate records from database
                    from MarketPulse.database.models import MarketData, Stock
                    from MarketPulse.config import LOOKBACK_DAYS, CANDLES_PER_DAY, WINDOW_CANDLES, REQUIRED_COLUMNS
                    import numpy as np
                    import pandas as pd
                    from MarketPulse.ml.dataset import build_and_verify_dataset
                    
                    stocks = db.query(Stock).all()
                    all_records = []
                    
                    for stock in stocks:
                        md_rows = db.query(MarketData).filter(MarketData.stock_id == stock.id).order_by(MarketData.timestamp.asc()).all()
                        if len(md_rows) < 150:
                            continue
                            
                        # Resample/group hourly closes into daily closes to calculate thresholds
                        df_md = pd.DataFrame([{
                            "Open": r.open, "High": r.high, "Low": r.low, "Close": r.close, "Volume": r.volume,
                            "DatetimeIST": r.timestamp
                        } for r in md_rows]).set_index("DatetimeIST")
                        
                        # Prepare context windows
                        # Create records list
                        from MarketPulse.extracted_code import build_sample_records
                        # we can reuse or implement sample records builder
                        # Let's call build_sample_records
                        records, _ = build_sample_records(stock.ticker, stock.name, stock.category, df_md)
                        all_records.extend(records)
                        
                    if all_records:
                        build_and_verify_dataset(all_records)
                        st.success("Successfully generated train/test candlestick splits!")
                    else:
                        st.error("No database records found. Run Ingestion (ETL) first to download price history.")
                        st.stop()
                
                # Run Training
                trainer = ModelTrainer(model_name=model_name)
                model, path = trainer.train(limit_epochs=epochs)
                st.success(f"Model trained and checkpoint saved at: {path}")
            except Exception as e:
                st.error(f"Training failed: {e}")
                
    st.markdown("---")
    
    st.subheader("🧪 Seed Mock Prediction Data")
    st.write("Seeds mock predictions into the database for visualization if models have not yet been trained.")
    
    if st.button("Seed Mock Predictions"):
        with st.spinner("Writing mock prediction scores..."):
            try:
                # Query stocks
                stocks = db.query(Stock).all()
                seed_count = 0
                for s in stocks:
                    # Let's get raw data closes
                    md_rows = db.query(MarketData).filter(MarketData.stock_id == s.id).order_by(MarketData.timestamp.desc()).limit(20).all()
                    if md_rows:
                        for idx, m in enumerate(md_rows):
                            # Generate a predicted label
                            pred_lbl = ["up", "down", "neutral"][idx % 3]
                            conf = 0.65 + (idx % 4) * 0.08
                            
                            # check if exists
                            existing = db.query(Prediction).filter(
                                Prediction.stock_id == s.id,
                                Prediction.prediction_timestamp == m.timestamp
                            ).first()
                            
                            if not existing:
                                pred = Prediction(
                                    stock_id=s.id,
                                    prediction_timestamp=m.timestamp,
                                    label_date=m.timestamp.date(),
                                    current_close=m.close,
                                    next_close=m.close * (1.0 + (idx % 3 - 1) * 0.015),
                                    next_day_return=(idx % 3 - 1) * 0.015,
                                    predicted_label=pred_lbl,
                                    true_label=pred_lbl,
                                    confidence=conf,
                                    correct=True
                                )
                                db.add(pred)
                                seed_count += 1
                db.commit()
                st.success(f"Successfully seeded {seed_count} prediction rows.")
            except Exception as e:
                db.rollback()
                st.error(f"Seeding failed: {e}")

db.close()
