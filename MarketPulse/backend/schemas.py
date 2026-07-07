from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date

# Stocks
class StockBase(BaseModel):
    ticker: str
    name: str
    category: str

class StockResponse(StockBase):
    id: int

    class Config:
        from_attributes = True

# Market Data
class MarketDataBase(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDataResponse(MarketDataBase):
    id: int
    stock_id: int

    class Config:
        from_attributes = True

# Technical Indicators
class TechnicalIndicatorResponse(BaseModel):
    id: int
    stock_id: int
    timestamp: datetime
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    rolling_volatility_20d: Optional[float] = None
    dynamic_threshold: Optional[float] = None
    doji: bool
    hammer: bool
    bullish_engulfing: bool
    bearish_engulfing: bool

    class Config:
        from_attributes = True

# Predictions
class PredictionResponse(BaseModel):
    id: int
    ticker: str
    stock_id: int
    prediction_timestamp: datetime
    label_date: date
    current_close: float
    next_close: float
    next_day_return: float
    predicted_label: str
    true_label: Optional[str] = None
    confidence: float
    correct: Optional[bool] = None

    class Config:
        from_attributes = True

# Data Quality Report
class DataQualityReportResponse(BaseModel):
    id: int
    timestamp: datetime
    run_id: str
    table_name: str
    rows_processed: int
    rows_skipped: int
    rows_rejected: int
    missing_values: int
    duplicate_count: int
    success_rate: float
    details: Optional[str] = None

    class Config:
        from_attributes = True

# Analytics Response
class AnalyticsSummaryResponse(BaseModel):
    top_gainers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    highest_volume: List[Dict[str, Any]]
    market_breadth: List[Dict[str, Any]]
    average_confidence: List[Dict[str, Any]]

# Model Run
class ModelRunResponse(BaseModel):
    id: int
    run_timestamp: datetime
    model_name: str
    config_name: str
    test_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float
    test_loss: float
    status: str
    checkpoint_path: str

    class Config:
        from_attributes = True

# Performance Response
class PerformanceSummaryResponse(BaseModel):
    metrics: Dict[str, Any]
    model_runs: List[ModelRunResponse]

# Refresh Response
class RefreshResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
