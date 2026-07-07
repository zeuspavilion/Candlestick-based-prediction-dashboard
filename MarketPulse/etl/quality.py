import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger("marketpulse.etl.quality")

class DataQualityValidator:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.required_columns = ["Open", "High", "Low", "Close", "Volume"]

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates the raw price DataFrame.
        Returns:
            - A cleaned DataFrame with invalid rows removed.
            - A Data Quality report summary dictionary.
        """
        if df.empty:
            return df, {
                "rows_processed": 0,
                "rows_skipped": 0,
                "rows_rejected": 0,
                "missing_values": 0,
                "duplicate_count": 0,
                "success_rate": 0.0,
                "details": "DataFrame is empty."
            }

        total_rows = len(df)
        details = {}
        
        # 1. Schema Validation (Check if all columns are present)
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            details["schema_errors"] = f"Missing columns: {missing_cols}"
            return pd.DataFrame(), {
                "rows_processed": total_rows,
                "rows_skipped": 0,
                "rows_rejected": total_rows,
                "missing_values": 0,
                "duplicate_count": 0,
                "success_rate": 0.0,
                "details": json.dumps(details)
            }

        # 2. Duplicate Detection (on Index/Timestamp)
        duplicate_mask = df.index.duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            details["duplicate_timestamps"] = [str(ts) for ts in df.index[duplicate_mask]]
            df = df[~duplicate_mask]

        # 3. Missing Values Check
        missing_mask = df[self.required_columns].isnull().any(axis=1)
        missing_values_count = int(df[self.required_columns].isnull().sum().sum())
        if missing_mask.any():
            details["missing_value_rows"] = [str(ts) for ts in df.index[missing_mask]]

        # Keep rows without missing values
        df_clean = df.dropna(subset=self.required_columns)

        # 4. Numeric validation & coercing
        for col in self.required_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna(subset=self.required_columns)

        # 5. Range Validation
        # Rules:
        # - Open, High, Low, Close, Volume must be positive (> 0)
        # - Low <= High
        # - Open <= High, Close <= High
        # - Low <= Open, Low <= Close
        initial_clean_count = len(df_clean)
        
        valid_prices = (
            (df_clean["Open"] > 0) & 
            (df_clean["High"] > 0) & 
            (df_clean["Low"] > 0) & 
            (df_clean["Close"] > 0) &
            (df_clean["Volume"] >= 0)
        )
        
        valid_ordering = (
            (df_clean["Low"] <= df_clean["High"]) &
            (df_clean["Open"] <= df_clean["High"]) &
            (df_clean["Close"] <= df_clean["High"]) &
            (df_clean["Low"] <= df_clean["Open"]) &
            (df_clean["Low"] <= df_clean["Close"])
        )
        
        valid_rows = valid_prices & valid_ordering
        rejected_rows_count = int((~valid_rows).sum())
        
        if rejected_rows_count > 0:
            details["price_range_errors"] = [str(ts) for ts in df_clean.index[~valid_rows]]
            df_clean = df_clean[valid_rows]

        # 6. Timestamp alignment checks (e.g. check if it's outside weekdays or trading hours)
        # Weekday check: 0=Monday, 4=Friday. 5 and 6 are weekend.
        weekend_mask = df_clean.index.dayofweek >= 5
        weekend_count = int(weekend_mask.sum())
        if weekend_count > 0:
            details["weekend_timestamps"] = [str(ts) for ts in df_clean.index[weekend_mask]]
            df_clean = df_clean[~weekend_mask]

        rows_kept = len(df_clean)
        rows_rejected = total_rows - rows_kept - duplicate_count
        
        # Success Rate
        success_rate = (rows_kept / total_rows) * 100.0 if total_rows > 0 else 0.0

        report = {
            "rows_processed": total_rows,
            "rows_skipped": duplicate_count,
            "rows_rejected": rows_rejected,
            "missing_values": missing_values_count,
            "duplicate_count": duplicate_count,
            "success_rate": success_rate,
            "details": json.dumps(details) if details else "All checks passed."
        }

        return df_clean, report
