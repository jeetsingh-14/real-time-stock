import pandas as pd
import yfinance as yf
import time
import logging
from sqlalchemy import create_engine
from twelvedata import TDClient

# === Logger setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# === Twelve Data setup ===
td = TDClient(apikey="47cbc897d102401db5efab4ebee0b8c1")  # Replace with your Twelve Data API key

# === Database connection ===
def connect_to_database():
    """
    Establish a connection to the MySQL database.
    Returns:
        SQLAlchemy engine
    """
    try:
        logger.info("Connecting to database...")
        engine = create_engine("mysql+pymysql://root:285project@127.0.0.1:3307/stock_sentiment")
        logger.info("Database connection established successfully")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# === Fetch stock data with fallback ===
def fetch_stock_data(symbol, start_date, end_date, max_retries=3, delay_sec=2):
    """
    Try fetching stock data from yfinance. If it fails or is rate limited, fall back to Twelve Data.
    
    Args:
        symbol (str): Stock ticker
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        max_retries (int): Max retries for yfinance
        delay_sec (int): Delay between retries

    Returns:
        pd.DataFrame with columns: ['timestamp', 'price_now']
    """
    # Try YFinance first
    for attempt in range(max_retries):
        try:
            logger.info(f"[YFinance] Fetching stock data for {symbol} (attempt {attempt+1})")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning(f"[YFinance] No data returned for {symbol}")
                break

            data.reset_index(inplace=True)
            data['price_now'] = data['Close']
            data.rename(columns={'Date': 'timestamp'}, inplace=True)
            return data[['timestamp', 'price_now']]

        except Exception as e:
            logger.warning(f"[YFinance] Error: {e}")
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait = delay_sec * (2 ** attempt)
                logger.warning(f"[YFinance] Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
                attempt += 1
            else:
                break

    # Fallback to Twelve Data
    try:
        logger.info(f"[TwelveData] Fallback: Fetching stock data for {symbol}")
        ts = td.time_series(
            symbol=symbol,
            interval="1day",
            start_date=start_date,
            end_date=end_date,
            outputsize=5000
        ).as_pandas()

        if ts.empty:
            logger.warning(f"[TwelveData] No data returned for {symbol}")
            return pd.DataFrame()

        ts.reset_index(inplace=True)
        ts.rename(columns={"datetime": "timestamp", "close": "price_now"}, inplace=True)
        return ts[["timestamp", "price_now"]]

    except Exception as e:
        logger.error(f"[TwelveData] Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
