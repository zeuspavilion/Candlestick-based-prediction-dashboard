import datetime
import logging
import json
import uuid
import time
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List, Optional
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import delete

from MarketPulse.config import (
    STOCKS,
    TICKERS_FLAT,
    YFINANCE_INTERVAL,
    YFINANCE_PERIOD,
    IST_TIMEZONE,
    MARKET_OPEN,
    MARKET_CLOSE,
    VOLATILITY_LOOKBACK_DAYS,
    VOLATILITY_MULTIPLIER,
    MIN_MOVEMENT_THRESHOLD,
    RAW_DIR,
)
from MarketPulse.database.connection import SessionLocal, engine
from MarketPulse.database.models import Stock, MarketData, TechnicalIndicator, PipelineLog, DataQualityReport
from MarketPulse.etl.quality import DataQualityValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marketpulse.etl.pipeline")

def ensure_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST_TIMEZONE)
    else:
        df.index = df.index.tz_convert(IST_TIMEZONE)
    df.index.name = "DatetimeIST"
    return df.sort_index()

def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    for level in range(df.columns.nlevels):
        if ticker in df.columns.get_level_values(level):
            return df.xs(ticker, axis=1, level=level)
    flattened = df.copy()
    flattened.columns = [col[0] for col in flattened.columns]
    return flattened

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series) -> tuple:
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_p = df["Open"]
    high_p = df["High"]
    low_p = df["Low"]
    close_p = df["Close"]
    
    body = (close_p - open_p).abs()
    candle_range = (high_p - low_p).clip(lower=1e-8)
    
    # Doji: body is <= 10% of total range
    df["doji"] = (body / candle_range) <= 0.10
    
    # Hammer check
    upper_wick = high_p - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - low_p
    
    df["hammer"] = (
        (lower_wick >= 2.0 * body.clip(lower=1e-8)) &
        (upper_wick <= 0.35 * body.clip(lower=1e-8)) &
        (df[["Open", "Close"]].max(axis=1) >= low_p + 0.60 * candle_range)
    )
    
    # Engulfing pattern (compares current with previous)
    df["bullish_engulfing"] = False
    df["bearish_engulfing"] = False
    
    for i in range(1, len(df)):
        prev_open = open_p.iloc[i-1]
        prev_close = close_p.iloc[i-1]
        curr_open = open_p.iloc[i]
        curr_close = close_p.iloc[i]
        
        # Bullish Engulfing
        if (prev_close < prev_open) and (curr_close > curr_open) and (curr_open <= prev_close) and (curr_close >= prev_open):
            df.loc[df.index[i], "bullish_engulfing"] = True
            
        # Bearish Engulfing
        if (prev_close > prev_open) and (curr_close < curr_open) and (curr_open >= prev_close) and (curr_close <= prev_open):
            df.loc[df.index[i], "bearish_engulfing"] = True
            
    return df

class ETLPipeline:
    def __init__(self, db_session=None):
        self.db = db_session or SessionLocal()

    def log_event(self, module: str, level: str, message: str, details: str = None):
        log = PipelineLog(module=module, level=level, message=message, details=details)
        self.db.add(log)
        self.db.commit()
        if level == "ERROR":
            logger.error(f"[{module}] {message}: {details}")
        elif level == "WARNING":
            logger.warning(f"[{module}] {message}")
        else:
            logger.info(f"[{module}] {message}")

    def fetch_raw_data(self, ticker: str, force_download: bool = False) -> pd.DataFrame:
        safe_name = ticker.replace(".", "_").replace("-", "_")
        csv_path = RAW_DIR / f"{safe_name}_{YFINANCE_INTERVAL}.csv"

        if csv_path.exists() and not force_download:
            self.log_event("ETL.Extract", "INFO", f"Loading {ticker} data from CSV cache")
            df = pd.read_csv(csv_path, parse_dates=["DatetimeIST"], index_col="DatetimeIST")
            return ensure_ist_index(df)

        self.log_event("ETL.Extract", "INFO", f"Downloading {ticker} from Yahoo Finance...")
        
        retries = 3
        for attempt in range(retries):
            try:
                df = yf.download(
                    tickers=ticker,
                    period=YFINANCE_PERIOD,
                    interval=YFINANCE_INTERVAL,
                    auto_adjust=False,
                    prepost=False,
                    progress=False,
                    threads=False,
                )
                if not df.empty:
                    break
            except Exception as e:
                self.log_event("ETL.Extract", "WARNING", f"Attempt {attempt+1} failed for {ticker}: {e}")
                time.sleep(2)
        else:
            raise ValueError(f"Failed to download data for {ticker} after {retries} attempts.")

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = flatten_yfinance_columns(df, ticker)
        df = ensure_ist_index(df)
        
        # Save cache
        df.to_csv(csv_path)
        self.log_event("ETL.Extract", "INFO", f"Successfully downloaded and cached {ticker} (rows: {len(df)})")
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_event("ETL.Transform", "INFO", f"Computing technical indicators...")
        
        df = df.copy()
        
        # 1. Moving Averages
        df["ma_20"] = df["Close"].rolling(20).mean()
        df["ma_50"] = df["Close"].rolling(50).mean()
        
        # 2. RSI
        df["rsi"] = calculate_rsi(df["Close"], period=14)
        
        # 3. MACD
        macd, macd_sig, macd_hist = calculate_macd(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = macd_sig
        df["macd_hist"] = macd_hist
        
        # 4. Candlestick Patterns
        df = detect_candlestick_patterns(df)
        
        # 5. Volatility & Dynamic Thresholds (Session based)
        session_df = df.between_time(MARKET_OPEN, MARKET_CLOSE).copy()
        daily_closes = session_df.groupby(session_df.index.date)["Close"].last()
        daily_returns = daily_closes.pct_change()
        
        rolling_vol = daily_returns.rolling(
            VOLATILITY_LOOKBACK_DAYS,
            min_periods=VOLATILITY_LOOKBACK_DAYS,
        ).std()
        
        dynamic_threshold = (VOLATILITY_MULTIPLIER * rolling_vol).clip(lower=MIN_MOVEMENT_THRESHOLD)
        
        # Map values back to hourly data index dates
        df["date_key"] = df.index.date
        
        vol_df = pd.DataFrame({
            "rolling_volatility_20d": rolling_vol,
            "dynamic_threshold": dynamic_threshold
        })
        vol_df.index.name = "date_key"
        
        df = df.join(vol_df, on="date_key")
        df = df.drop(columns=["date_key"])
        
        return df

    def load_stock_data(self, stock_id: int, ticker: str, df: pd.DataFrame):
        self.log_event("ETL.Load", "INFO", f"Loading market data & indicators for {ticker} to DB")
        
        # SQLite vs Postgres check
        is_sqlite = engine.dialect.name == "sqlite"
        
        # 1. Save Market Data
        market_data_records = []
        indicator_records = []
        
        for ts, row in df.iterrows():
            ts_native = ts.to_pydatetime().replace(tzinfo=None) # remove tz for database standard datetime
            
            market_data_records.append({
                "stock_id": stock_id,
                "timestamp": ts_native,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })
            
            indicator_records.append({
                "stock_id": stock_id,
                "timestamp": ts_native,
                "rsi": float(row["rsi"]) if not pd.isna(row["rsi"]) else None,
                "macd": float(row["macd"]) if not pd.isna(row["macd"]) else None,
                "macd_signal": float(row["macd_signal"]) if not pd.isna(row["macd_signal"]) else None,
                "macd_hist": float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else None,
                "ma_20": float(row["ma_20"]) if not pd.isna(row["ma_20"]) else None,
                "ma_50": float(row["ma_50"]) if not pd.isna(row["ma_50"]) else None,
                "rolling_volatility_20d": float(row["rolling_volatility_20d"]) if not pd.isna(row["rolling_volatility_20d"]) else None,
                "dynamic_threshold": float(row["dynamic_threshold"]) if not pd.isna(row["dynamic_threshold"]) else None,
                "doji": bool(row["doji"]),
                "hammer": bool(row["hammer"]),
                "bullish_engulfing": bool(row["bullish_engulfing"]),
                "bearish_engulfing": bool(row["bearish_engulfing"]),
            })
            
        # Bulk Upsert
        # Since SQLite does not have simple standard upsert in early SQLAlchemy versions easily, 
        # for SQLite we'll delete and re-insert or use connection.execute. Deleting and bulk inserting 
        # is extremely reliable and safe for both SQLite and Postgres.
        if is_sqlite:
            # Delete old data
            self.db.execute(delete(MarketData).where(MarketData.stock_id == stock_id))
            self.db.execute(delete(TechnicalIndicator).where(TechnicalIndicator.stock_id == stock_id))
            self.db.commit()
            
            # Bulk Insert
            self.db.bulk_insert_mappings(MarketData, market_data_records)
            self.db.bulk_insert_mappings(TechnicalIndicator, indicator_records)
            self.db.commit()
        else:
            # Postgres native UPSERT
            # 1. Market Data
            stmt_md = pg_insert(MarketData).values(market_data_records)
            update_dict_md = {
                c.name: c for c in stmt_md.excluded if c.name not in ["id", "stock_id", "timestamp"]
            }
            stmt_md = stmt_md.on_conflict_do_update(
                constraint="uq_stock_market_timestamp",
                set_=update_dict_md
            )
            self.db.execute(stmt_md)
            
            # 2. Technical Indicators
            stmt_ti = pg_insert(TechnicalIndicator).values(indicator_records)
            update_dict_ti = {
                c.name: c for c in stmt_ti.excluded if c.name not in ["id", "stock_id", "timestamp"]
            }
            stmt_ti = stmt_ti.on_conflict_do_update(
                constraint="uq_stock_indicator_timestamp",
                set_=update_dict_ti
            )
            self.db.execute(stmt_ti)
            self.db.commit()
            
        self.log_event("ETL.Load", "INFO", f"Loaded {len(df)} rows for {ticker} successfully")

    def run(self, ticker: Optional[str] = None, force_download: bool = False):
        """Runs the ETL pipeline for a single stock or all stocks."""
        tickers = [ticker] if ticker else TICKERS_FLAT
        run_id = str(uuid.uuid4())
        
        self.log_event("ETL.Pipeline", "INFO", f"Starting ETL Run {run_id} for {len(tickers)} stocks")
        
        for symbol in tickers:
            try:
                # 1. Fetch DB Stock object to link FK
                stock = self.db.query(Stock).filter(Stock.ticker == symbol).first()
                if not stock:
                    self.log_event("ETL.Pipeline", "ERROR", f"Ticker {symbol} not found in database. Seed the database first.")
                    continue
                
                # 2. Extract Stage
                df_raw = self.fetch_raw_data(symbol, force_download=force_download)
                
                # 3. Data Quality Stage
                dq = DataQualityValidator(symbol)
                df_clean, report = dq.validate(df_raw)
                
                # Save Data Quality Report
                dq_report = DataQualityReport(
                    run_id=run_id,
                    table_name="market_data",
                    rows_processed=report["rows_processed"],
                    rows_skipped=report["rows_skipped"],
                    rows_rejected=report["rows_rejected"],
                    missing_values=report["missing_values"],
                    duplicate_count=report["duplicate_count"],
                    success_rate=report["success_rate"],
                    details=report["details"]
                )
                self.db.add(dq_report)
                self.db.commit()
                
                if df_clean.empty:
                    self.log_event("ETL.Pipeline", "WARNING", f"No clean data remaining for {symbol} after DQ validation.")
                    continue
                
                # 4. Transform Stage
                df_features = self.compute_indicators(df_clean)
                
                # 5. Load Stage
                self.load_stock_data(stock.id, symbol, df_features)
                
            except Exception as e:
                self.log_event("ETL.Pipeline", "ERROR", f"ETL Pipeline failed for {symbol}: {e}", details=str(e))
                self.db.rollback()
                
        self.log_event("ETL.Pipeline", "INFO", f"Finished ETL Run {run_id}")

    def close(self):
        self.db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MarketPulse ETL Pipeline")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker to ingest (default: all)")
    parser.add_argument("--force", action="store_true", help="Force download from yfinance")
    args = parser.parse_args()

    pipeline = ETLPipeline()
    try:
        pipeline.run(ticker=args.ticker, force_download=args.force)
    finally:
        pipeline.close()
