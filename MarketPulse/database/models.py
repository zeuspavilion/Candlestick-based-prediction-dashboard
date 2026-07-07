import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Date,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from MarketPulse.database.connection import Base

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)

    # Relationships
    market_data = relationship("MarketData", back_populates="stock", cascade="all, delete-orphan")
    indicators = relationship("TechnicalIndicator", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    watchlist = relationship("Watchlist", back_populates="stock", uselist=False, cascade="all, delete-orphan")


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Relationship
    stock = relationship("Stock", back_populates="market_data")

    # Constraints
    __table_args__ = (
        UniqueConstraint("stock_id", "timestamp", name="uq_stock_market_timestamp"),
    )


class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    ma_20 = Column(Float, nullable=True)
    ma_50 = Column(Float, nullable=True)
    rolling_volatility_20d = Column(Float, nullable=True)
    dynamic_threshold = Column(Float, nullable=True)
    doji = Column(Boolean, default=False)
    hammer = Column(Boolean, default=False)
    bullish_engulfing = Column(Boolean, default=False)
    bearish_engulfing = Column(Boolean, default=False)

    # Relationship
    stock = relationship("Stock", back_populates="indicators")

    # Constraints
    __table_args__ = (
        UniqueConstraint("stock_id", "timestamp", name="uq_stock_indicator_timestamp"),
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    model_name = Column(String, nullable=False)
    config_name = Column(String, nullable=False)
    test_accuracy = Column(Float, nullable=False)
    test_macro_f1 = Column(Float, nullable=False)
    test_weighted_f1 = Column(Float, nullable=False)
    test_loss = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=False)

    # Relationships
    predictions = relationship("Prediction", back_populates="model_run")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    label_date = Column(Date, nullable=False)
    current_close = Column(Float, nullable=False)
    next_close = Column(Float, nullable=False)
    next_day_return = Column(Float, nullable=False)
    predicted_label = Column(String, nullable=False)  # "up", "down", "neutral"
    true_label = Column(String, nullable=True)       # "up", "down", "neutral"
    confidence = Column(Float, nullable=False)
    correct = Column(Boolean, nullable=True)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    model_run = relationship("ModelRun", back_populates="predictions")


class PipelineLog(Base):
    __tablename__ = "pipeline_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    module = Column(String, nullable=False, index=True)
    level = Column(String, nullable=False, index=True)
    message = Column(String, nullable=False)
    details = Column(Text, nullable=True)


class DataQualityReport(Base):
    __tablename__ = "data_quality_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    run_id = Column(String, nullable=False, index=True)
    table_name = Column(String, nullable=False, index=True)
    rows_processed = Column(Integer, nullable=False)
    rows_skipped = Column(Integer, nullable=False)
    rows_rejected = Column(Integer, nullable=False)
    missing_values = Column(Integer, nullable=False)
    duplicate_count = Column(Integer, nullable=False)
    success_rate = Column(Float, nullable=False)
    details = Column(Text, nullable=True)  # JSON-formatted string of failures


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), unique=True, nullable=False)
    added_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationship
    stock = relationship("Stock", back_populates="watchlist")
