"""
Merge sentiment data with stock price data and save to merged_price_sentiment table.
"""

import logging
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import text
from stockmarket_analysis.backend.utils.config import TICKERS
from stockmarket_analysis.backend.utils.data_fetcher import connect_to_database, fetch_stock_data

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
log_dir = os.path.join(Path(__file__).resolve().parents[2], "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pipeline.log")
fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
logger.addHandler(fh)

def load_sentiment_data():
    try:
        logger.info("Loading sentiment data from database")
        engine = connect_to_database()
        query = """
        SELECT * FROM combined_sentiment_data
        WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        """
        sentiment_df = pd.read_sql(query, engine)
        sentiment_df.rename(columns={'symbol': 'ticker', 'sentiment': 'sentiment_score'}, inplace=True)
        logger.info(f"Loaded {len(sentiment_df)} sentiment records")
        return sentiment_df
    except Exception as e:
        logger.error(f"Error loading sentiment data: {str(e)}")
        return pd.DataFrame()

def merge_data(sentiment_df, price_df, symbol):
    try:
        if sentiment_df.empty or price_df.empty:
            logger.warning(f"Empty dataframe detected for {symbol}. Skipping merge.")
            return pd.DataFrame()

        symbol_sentiment = sentiment_df[sentiment_df['ticker'] == symbol].copy()
        if symbol_sentiment.empty:
            logger.warning(f"No sentiment data found for {symbol}. Skipping merge.")
            return pd.DataFrame()

        symbol_sentiment['timestamp'] = pd.to_datetime(symbol_sentiment['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        symbol_sentiment['date'] = symbol_sentiment['timestamp'].dt.date
        price_df['date'] = price_df['timestamp'].dt.date

        merged_df = pd.merge(
            symbol_sentiment,
            price_df,
            on='date',
            how='inner',
            suffixes=('_sentiment', '_price')
        )

        if merged_df.empty:
            logger.warning(f"No matching dates found for {symbol}. Skipping.")
            return pd.DataFrame()

        merged_df['ticker'] = symbol
        merged_df.rename(columns={'timestamp_sentiment': 'timestamp'}, inplace=True)
        final_df = merged_df[['ticker', 'timestamp', 'sentiment_score', 'price_now', 'date']]
        logger.info(f"Successfully merged {len(final_df)} records for {symbol}")
        return final_df
    except Exception as e:
        logger.error(f"Error merging data for {symbol}: {str(e)}")
        return pd.DataFrame()

def save_to_database(df):
    try:
        if df.empty:
            logger.warning("Empty dataframe. Nothing to save to database.")
            return

        engine = connect_to_database()

        # Match column names to DB
        df.rename(columns={
            'ticker': 'symbol',
            'sentiment_score': 'sentiment'
        }, inplace=True)

        # Keep the original timestamp (do not overwrite)
        df.drop(columns=['date'], inplace=True)

        df = df[['symbol', 'timestamp', 'sentiment', 'price_now']]

        df.to_sql('merged_price_sentiment', engine, if_exists='append', index=False)
        logger.info(f"Successfully saved {len(df)} records to database")

    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")

def main():
    try:
        logger.info("Starting sentiment and price data merging process")
        sentiment_df = load_sentiment_data()
        if sentiment_df.empty:
            logger.error("No sentiment data available. Exiting.")
            return

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        all_merged_data = []

        for idx, symbol in enumerate(TICKERS, start=1):
            logger.info(f"Processing ticker {idx}/{len(TICKERS)}: {symbol}")
            price_df = fetch_stock_data(symbol, start_date, end_date)
            if price_df.empty:
                logger.warning(f"No price data found for {symbol}. Skipping.")
                continue
            merged_data = merge_data(sentiment_df, price_df, symbol)
            if not merged_data.empty:
                all_merged_data.append(merged_data)

        if all_merged_data:
            final_df = pd.concat(all_merged_data, ignore_index=True)
            save_to_database(final_df)
            logger.info(f"Successfully merged and saved {len(final_df)} records from {len(TICKERS)} tickers.")
        else:
            logger.warning("No data was merged. Nothing to save.")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
