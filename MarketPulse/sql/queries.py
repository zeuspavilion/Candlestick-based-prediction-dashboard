from sqlalchemy import text

# Dictionary of raw SQL queries representing Module 2: SQL Analytics
ANALYTICAL_QUERIES = {
    # 1. Top Gainers of the last recorded day
    "top_gainers": """
        WITH LatestDates AS (
            SELECT stock_id, DATE(MAX(timestamp)) as max_date
            FROM market_data
            GROUP BY stock_id
        ),
        DailyPrices AS (
            SELECT 
                md.stock_id,
                DATE(md.timestamp) as price_date,
                MIN(md.open) as day_open,
                MAX(md.close) as day_close
            FROM market_data md
            GROUP BY md.stock_id, DATE(md.timestamp)
        )
        SELECT 
            s.ticker,
            s.name,
            s.category,
            dp.price_date,
            dp.day_open,
            dp.day_close,
            ((dp.day_close - dp.day_open) / dp.day_open) * 100.0 as pct_change
        FROM DailyPrices dp
        JOIN LatestDates ld ON dp.stock_id = ld.stock_id AND dp.price_date = ld.max_date
        JOIN stocks s ON s.id = dp.stock_id
        ORDER BY pct_change DESC
        LIMIT :limit;
    """,

    # 2. Top Losers of the last recorded day
    "top_losers": """
        WITH LatestDates AS (
            SELECT stock_id, DATE(MAX(timestamp)) as max_date
            FROM market_data
            GROUP BY stock_id
        ),
        DailyPrices AS (
            SELECT 
                md.stock_id,
                DATE(md.timestamp) as price_date,
                MIN(md.open) as day_open,
                MAX(md.close) as day_close
            FROM market_data md
            GROUP BY md.stock_id, DATE(md.timestamp)
        )
        SELECT 
            s.ticker,
            s.name,
            s.category,
            dp.price_date,
            dp.day_open,
            dp.day_close,
            ((dp.day_close - dp.day_open) / dp.day_open) * 100.0 as pct_change
        FROM DailyPrices dp
        JOIN LatestDates ld ON dp.stock_id = ld.stock_id AND dp.price_date = ld.max_date
        JOIN stocks s ON s.id = dp.stock_id
        ORDER BY pct_change ASC
        LIMIT :limit;
    """,

    # 3. Highest Volume Stocks
    "highest_volume": """
        WITH LatestDates AS (
            SELECT stock_id, DATE(MAX(timestamp)) as max_date
            FROM market_data
            GROUP BY stock_id
        )
        SELECT 
            s.ticker,
            s.name,
            DATE(md.timestamp) as trade_date,
            SUM(md.volume) as total_volume
        FROM market_data md
        JOIN LatestDates ld ON md.stock_id = ld.stock_id AND DATE(md.timestamp) = ld.max_date
        JOIN stocks s ON s.id = md.stock_id
        GROUP BY s.ticker, s.name, DATE(md.timestamp)
        ORDER BY total_volume DESC
        LIMIT :limit;
    """,

    # 4. Average Daily Return per Ticker
    "avg_daily_return": """
        WITH DailyPrices AS (
            SELECT 
                stock_id,
                DATE(timestamp) as price_date,
                (MAX(close) - MIN(open)) / MIN(open) as daily_ret
            FROM market_data
            GROUP BY stock_id, DATE(timestamp)
        )
        SELECT 
            s.ticker,
            s.name,
            AVG(dp.daily_ret) * 100.0 as avg_return_pct,
            COUNT(dp.price_date) as trading_days
        FROM DailyPrices dp
        JOIN stocks s ON s.id = dp.stock_id
        GROUP BY s.ticker, s.name
        ORDER BY avg_return_pct DESC;
    """,

    # 5. Average Rolling Volatility (20-day) per Stock Category
    "rolling_volatility": """
        SELECT 
            s.category,
            AVG(ti.rolling_volatility_20d) * 100.0 as avg_volatility_pct
        FROM technical_indicators ti
        JOIN stocks s ON s.id = ti.stock_id
        WHERE ti.rolling_volatility_20d IS NOT NULL
        GROUP BY s.category
        ORDER BY avg_volatility_pct DESC;
    """,

    # 6. Overall Prediction Accuracy
    "prediction_accuracy": """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN correct = 1 OR correct = 'true' OR correct = true THEN 1 ELSE 0 END) as correct_predictions,
            (CAST(SUM(CASE WHEN correct = 1 OR correct = 'true' OR correct = true THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)) * 100.0 as accuracy_pct
        FROM predictions
        WHERE true_label IS NOT NULL;
    """,

    # 7. Prediction Label Distribution
    "prediction_distribution": """
        SELECT 
            predicted_label,
            COUNT(*) as count,
            (CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM predictions)) * 100.0 as percentage
        FROM predictions
        GROUP BY predicted_label
        ORDER BY count DESC;
    """,

    # 8. Most Active Stocks (highest total transaction days recorded)
    "most_active_stocks": """
        SELECT 
            s.ticker,
            s.name,
            s.category,
            COUNT(md.id) as hourly_records_count,
            MIN(md.timestamp) as tracking_started,
            MAX(md.timestamp) as tracking_ended
        FROM market_data md
        JOIN stocks s ON s.id = md.stock_id
        GROUP BY s.ticker, s.name, s.category
        ORDER BY hourly_records_count DESC;
    """,

    # 9. Latest Model Predictions
    "latest_predictions": """
        WITH RankedPredictions AS (
            SELECT 
                p.id,
                p.stock_id,
                p.prediction_timestamp,
                p.label_date,
                p.current_close,
                p.predicted_label,
                p.confidence,
                ROW_NUMBER() OVER(PARTITION BY p.stock_id ORDER BY p.prediction_timestamp DESC) as rn
            FROM predictions p
        )
        SELECT 
            s.ticker,
            s.name,
            rp.prediction_timestamp,
            rp.label_date,
            rp.current_close,
            rp.predicted_label,
            rp.confidence
        FROM RankedPredictions rp
        JOIN stocks s ON s.id = rp.stock_id
        WHERE rp.rn = 1
        ORDER BY rp.confidence DESC;
    """,

    # 10. Historical Prediction Accuracy by Ticker
    "historical_accuracy_by_ticker": """
        SELECT 
            s.ticker,
            s.name,
            COUNT(p.id) as total_predictions,
            SUM(CASE WHEN p.correct = 1 OR p.correct = 'true' OR p.correct = true THEN 1 ELSE 0 END) as correct_count,
            (CAST(SUM(CASE WHEN p.correct = 1 OR p.correct = 'true' OR p.correct = true THEN 1 ELSE 0 END) AS FLOAT) / COUNT(p.id)) * 100.0 as accuracy_pct
        FROM predictions p
        JOIN stocks s ON s.id = p.stock_id
        WHERE p.true_label IS NOT NULL
        GROUP BY s.ticker, s.name
        ORDER BY accuracy_pct DESC;
    """,

    # 11. Market Breadth (percentage of stocks predicted to go UP vs DOWN)
    "market_breadth": """
        WITH LatestPredictions AS (
            SELECT 
                stock_id,
                predicted_label,
                ROW_NUMBER() OVER(PARTITION BY stock_id ORDER BY prediction_timestamp DESC) as rn
            FROM predictions
        )
        SELECT 
            predicted_label,
            COUNT(*) as count,
            (CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM LatestPredictions WHERE rn = 1)) * 100.0 as pct_of_market
        FROM LatestPredictions
        WHERE rn = 1
        GROUP BY predicted_label;
    """,

    # 12. Average Confidence Level by Predicted Label
    "average_confidence": """
        SELECT 
            predicted_label,
            AVG(confidence) * 100.0 as avg_confidence_pct,
            COUNT(*) as sample_size
        FROM predictions
        GROUP BY predicted_label
        ORDER BY avg_confidence_pct DESC;
    """,

    # 13. Model Performance History across different ModelRuns
    "model_performance_history": """
        SELECT 
            id as run_id,
            run_timestamp,
            model_name,
            config_name,
            test_accuracy * 100.0 as accuracy_pct,
            test_macro_f1 * 100.0 as macro_f1_pct,
            test_weighted_f1 * 100.0 as weighted_f1_pct,
            test_loss
        FROM model_runs
        ORDER BY run_timestamp DESC;
    """
}

def execute_analytical_query(db_session, query_name: str, params: dict = None) -> list:
    """Executes a predefined analytical query and returns records as list of dicts."""
    query_str = ANALYTICAL_QUERIES.get(query_name)
    if not query_str:
        raise ValueError(f"Analytical query '{query_name}' not defined.")
        
    p = params or {}
    # Use SQLAlchemy text execution
    result = db_session.execute(text(query_str), p)
    
    # Check if the query has return rows (e.g. SELECT)
    if result.returns_rows:
        keys = result.keys()
        return [dict(zip(keys, row)) for row in result.fetchall()]
    return []
