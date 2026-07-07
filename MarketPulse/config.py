import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_hourly"
IMAGE_DIR = DATA_DIR / "candlestick_images_multiclass"
METADATA_PATH = DATA_DIR / "metadata_multiclass.csv"
SUMMARY_PATH = DATA_DIR / "dataset_summary_multiclass.json"
MANIFEST_DIR = DATA_DIR / "manifests_multiclass"
CLASS_MAPPING_PATH = DATA_DIR / "class_to_idx_multiclass.json"
MODEL_DIR = DATA_DIR / "models"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DIR, IMAGE_DIR, MANIFEST_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
# Default to SQLite local database file for ease of verification, but Docker Compose overrides this to PostgreSQL.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/marketpulse.db")

# Market Parameters
IST_TIMEZONE = "Asia/Kolkata"
YFINANCE_PERIOD = "2y"
YFINANCE_INTERVAL = "1h"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
CANDLES_PER_DAY = 6
LOOKBACK_DAYS = 3
WINDOW_CANDLES = CANDLES_PER_DAY * LOOKBACK_DAYS

# Model labeling parameters
VOLATILITY_LOOKBACK_DAYS = 20
VOLATILITY_MULTIPLIER = 0.5
MIN_MOVEMENT_THRESHOLD = 0.003
LABELS = ["down", "neutral", "up"]
CLASS_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_CLASS = {idx: label for label, idx in CLASS_TO_IDX.items()}

# Dataset Split
SPLIT_RATIOS = {"train": 0.70, "validation": 0.20, "test": 0.10}
PURGE_SAMPLES_BETWEEN_SPLITS = LOOKBACK_DAYS - 1
RANDOM_SEED = 42

# Image generation parameters
IMAGE_SIZE = (224, 224)
IMAGE_DPI = 100

# STOCKS List
STOCKS = {
    "Private Banks": {
        "HDFC Bank Ltd": "HDFCBANK.NS",
        "ICICI Bank Ltd": "ICICIBANK.NS",
        "Axis Bank Ltd": "AXISBANK.NS",
        "Kotak Mahindra Bank Ltd": "KOTAKBANK.NS",
        "IndusInd Bank Ltd": "INDUSINDBK.NS",
    },
    "PSU Banks": {
        "State Bank of India": "SBIN.NS",
        "Bank of Baroda": "BANKBARODA.NS",
        "Punjab National Bank": "PNB.NS",
        "Canara Bank": "CANBK.NS",
        "Union Bank of India": "UNIONBANK.NS",
    },
    "Others": {
        "Federal Bank Ltd": "FEDERALBNK.NS",
        "IDFC First Bank Ltd": "IDFCFIRSTB.NS",
        "Bandhan Bank Ltd": "BANDHANBNK.NS",
        "RBL Bank Ltd": "RBLBANK.NS",
        "AU Small Finance Bank Ltd": "AUBANK.NS",
    },
}

TICKERS_FLAT = [ticker for group in STOCKS.values() for ticker in group.values()]
TICKERS_TO_NAME = {ticker: name for group in STOCKS.values() for name, ticker in group.items()}
TICKERS_TO_CATEGORY = {ticker: cat for cat, group in STOCKS.items() for ticker in group.values()}
