import logging
from MarketPulse.database.connection import engine, Base, SessionLocal
from MarketPulse.database.models import Stock, Watchlist
from MarketPulse.config import STOCKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marketpulse.init_db")

def init_database():
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

    db = SessionLocal()
    try:
        # Seed stocks table
        logger.info("Seeding stocks table...")
        seeded_count = 0
        watchlist_seeded = 0

        for category, members in STOCKS.items():
            for name, ticker in members.items():
                existing = db.query(Stock).filter(Stock.ticker == ticker).first()
                if not existing:
                    stock = Stock(ticker=ticker, name=name, category=category)
                    db.add(stock)
                    db.flush()  # Populate stock.id
                    seeded_count += 1

                    # Add HDFC and SBI to watchlist by default
                    if ticker in ["HDFCBANK.NS", "SBIN.NS"]:
                        watchlist_entry = Watchlist(stock_id=stock.id)
                        db.add(watchlist_entry)
                        watchlist_seeded += 1
                else:
                    # Make sure watchlist has HDFC and SBI even if already seeded
                    if ticker in ["HDFCBANK.NS", "SBIN.NS"]:
                        existing_wl = db.query(Watchlist).filter(Watchlist.stock_id == existing.id).first()
                        if not existing_wl:
                            watchlist_entry = Watchlist(stock_id=existing.id)
                            db.add(watchlist_entry)
                            watchlist_seeded += 1

        db.commit()
        logger.info(f"Database seeding completed. Seeded {seeded_count} stocks, {watchlist_seeded} default watchlist items.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding database: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
