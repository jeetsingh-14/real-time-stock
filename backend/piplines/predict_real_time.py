
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from twelvedata import TDClient

# Import config from utils
from stockmarket_analysis.backend.utils.config import TICKERS

# === CONFIG ===
tickers = TICKERS
period = "7d"
interval = "1h"
label_horizon = 3  # Predict 3 hours ahead

# Use dynamic path for model output
model_dir = os.path.join(Path(__file__).resolve().parents[2], "model")
os.makedirs(model_dir, exist_ok=True)
model_output_path = os.path.join(model_dir, "tech_model.pkl")

# === Twelve Data setup ===
td_client = TDClient(apikey="47cbc897d102401db5efab4ebee0b8c1")  # Replace with your Twelve Data API key

def fetch_price_data(symbol, td_client, period="7d", interval="1h"):
    """Fetch stock price data using yfinance first, then Twelve Data as fallback."""

    # Try YFinance first
    try:
        print(f"[YFinance] Downloading {symbol}...")
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df['timestamp'] = pd.to_datetime(df['Date']) if 'Date' in df.columns else df['Datetime']
            df['price_now'] = df['Close']
            return df[['timestamp', 'price_now']]
        else:
            print(f"[YFinance] No data for {symbol}, trying Twelve Data...")
    except Exception as e:
        print(f"[YFinance] Error for {symbol}: {e}")

    # Fallback: Twelve Data
    try:
        print(f"[TwelveData] Downloading {symbol}...")
        ts = td_client.time_series(
            symbol=symbol,
            interval="1h",
            outputsize=5000
        ).as_pandas()

        if ts.empty:
            print(f"[TwelveData] No data returned for {symbol}")
            return pd.DataFrame()

        print(f"[TwelveData] Columns returned for {symbol}: {list(ts.columns)}")

        # Reset and rename

        ts.reset_index(inplace=True)
        ts.columns = [col.lower() for col in ts.columns]  # lowercase all for safety

        if 'datetime' in ts.columns:
            ts.rename(columns={'datetime': 'timestamp'}, inplace=True)

        if 'close' not in ts.columns:
            print(f"Warning: No 'close' column for {symbol}")
            return pd.DataFrame()

        ts.rename(columns={'close': 'price_now'}, inplace=True)
        return ts[['timestamp', 'price_now']]

    except Exception as e:
        print(f"[TwelveData] Error fetching data for {symbol}: {e}")
        if "API credits" in str(e) or "limit" in str(e).lower():
            print("Sleeping for 30 seconds due to API limit...")
            time.sleep(15)
        return pd.DataFrame()

def generate_features_and_labels(ticker, td_client):
    df = fetch_price_data(ticker, td_client)

    if df.empty:
        print(f"Warning: No data for {ticker}")
        return pd.DataFrame()

    # Handle multi-index if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    close_cols = [col for col in df.columns if 'close' in col.lower() or col == 'Close']
    if not close_cols:
        print(f"Warning: No close column for {ticker}")
        return pd.DataFrame()

    close_col = close_cols[0]
    df['Close'] = df[close_col]  # Normalize to 'Close'

    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Target label
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
    all_data = []

    for ticker in tickers:
        df = generate_features_and_labels(ticker, td_client)
        if not df.empty:
            all_data.append(df)
        time.sleep(15)  # Throttle requests to avoid API rate limits

    if not all_data:
        print("No data collected. Exiting.")
        return

    dataset = pd.concat(all_data, ignore_index=True)

    X = dataset[['returns', 'volatility', 'sma_20', 'ema_10', 'Close']]
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    print("Training technical fallback model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Evaluation:\n", classification_report(y_test, y_pred, target_names=["DOWN", "NEUTRAL", "UP"]))

    joblib.dump(model, model_output_path)
    print(f"Technical fallback model saved as {model_output_path}")

if __name__ == "__main__":
    main()
