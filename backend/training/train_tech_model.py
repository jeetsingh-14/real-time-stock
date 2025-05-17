import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
import os
import logging
from pathlib import Path
from twelvedata import TDClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from stockmarket_analysis.backend.utils.config import TICKERS

# Configure logging
log_dir = os.path.join(Path(__file__).resolve().parents[2], "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_tech_model.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIG ===
tickers = TICKERS
period = "7d"
interval = "1h"
label_horizon = 3  # Predict 3 hours ahead

# Use dynamic path for model output
model_dir = os.path.join(Path(__file__).resolve().parents[2], "model")
os.makedirs(model_dir, exist_ok=True)
model_output_path = os.path.join(model_dir, "tech_model.pkl")

td = TDClient(apikey="47cbc897d102401db5efab4ebee0b8c1")  # Replace with your Twelve Data key

def fetch_data_with_fallback(ticker):
    try:
        logger.info(f"[YFinance] Downloading {ticker}...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if not df.empty:
            df.reset_index(inplace=True)
            df['Close'] = df['Close']
            df['timestamp'] = df['Datetime'] if 'Datetime' in df.columns else df['Date']
            return df[['timestamp', 'Close']]

        logger.warning(f"[YFinance] No data, trying Twelve Data for {ticker}...")
    except Exception as e:
        logger.error(f"[YFinance] Error for {ticker}: {e}")

    # Delay to respect API rate limit
    logger.info(f"[TwelveData] Sleeping 10 seconds before fetching {ticker}...")
    time.sleep(10)

    try:
        ts = td.time_series(
            symbol=ticker,
            interval="1h",
            outputsize=5000
        ).as_pandas()

        if ts.empty:
            logger.warning(f"[TwelveData] No data for {ticker}")
            return pd.DataFrame()

        ts.reset_index(inplace=True)
        ts.rename(columns={"datetime": "timestamp", "close": "Close"}, inplace=True)
        ts['Close'] = pd.to_numeric(ts['Close'], errors='coerce')
        ts.dropna(subset=['Close'], inplace=True)
        return ts[['timestamp', 'Close']]

    except Exception as e:
        logger.error(f"[TwelveData] Error for {ticker}: {e}")
        return pd.DataFrame()

def generate_features_and_labels(ticker):
    df = fetch_data_with_fallback(ticker)
    if df.empty:
        logger.warning(f"No data for {ticker}")
        return pd.DataFrame()

    # Sort & calculate indicators
    df.sort_values('timestamp', inplace=True)
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Label
    df['future_close'] = df['Close'].shift(-label_horizon)
    df['price_change'] = df['future_close'] - df['Close']

    def classify(change):
        if change > 0.5:
            return 2  # UP
        elif change < -0.5:
            return 0  # DOWN
        else:
            return 1  # NEUTRAL

    df['label'] = df['price_change'].apply(classify)
    df = df.dropna()

    return df[['returns', 'volatility', 'sma_20', 'ema_10', 'Close', 'label']]

def main():
    logger.info("Starting technical model training process")
    all_data = []

    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        df = generate_features_and_labels(ticker)
        if not df.empty:
            all_data.append(df)
            logger.info(f"Added {len(df)} records for {ticker}")

    if not all_data:
        logger.error("No data collected. Exiting.")
        return

    dataset = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total dataset size: {len(dataset)} records")

    X = dataset[['returns', 'volatility', 'sma_20', 'ema_10', 'Close']]
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    logger.info("Training technical fallback model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "NEUTRAL", "UP"])
    logger.info(f"\nModel Evaluation:\n{report}")

    joblib.dump(model, model_output_path)
    logger.info(f"Technical fallback model saved as {model_output_path}")
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()
