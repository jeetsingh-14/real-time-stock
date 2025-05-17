import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import time
import os
import logging
from pathlib import Path
from stockmarket_analysis.backend.utils.config import MYSQL_CONNECTION_STRING, TICKERS

# Configure logging
log_dir = os.path.join(Path(__file__).resolve().parents[2], "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "sentiment_data.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === API KEYS ===
NEWSDATA_API_KEY = "pub_794245cdb107a3bc95494420f64ddf03606bc"
GNEWS_API_KEYS = [
    "ea31b49b4d57903b2e9e7f569ec9ae30",
    "53ce217894164aca545cf8337da6624b",
    "384b201a52e74e670be64450e99c71c2",
    "f24ebae87c0395f24b949fc698eff423",
    "de56656b5832fc1534e6f3fe472c377d",
    "c411f74ca32749cc2425974710f68889",
    "5d6ea6a4cae88e687adf289ce2282f27",
    "58a1edc3b5164b830ec50b945a7392cb",
    "cd74a28f7bc23ac076b395c224a03c43",
    "f66becd3d5408bb6179a10fb478f121c",
    "92b5ccaf376e5b7a9b946ac3da6c5627"
]

# === Tickers to monitor ===
ticker_list = TICKERS

# === Sentiment Analyzer ===
analyzer = SentimentIntensityAnalyzer()

# === MySQL Engine ===
engine = create_engine(MYSQL_CONNECTION_STRING)

# === Ensure symbol column exists ===
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE combined_sentiment_data ADD COLUMN symbol VARCHAR(10);"))
        logger.info("'symbol' column added to combined_sentiment_data.")
except Exception as e:
    logger.info(f"Skipping column creation (likely already exists): {e}")

# === Clean and deduplicate logic ===
def clean_df(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Optional: Remove obvious junk
    df = df[df["text"].str.len() > 20]

    # Clean text for deduplication
    df["text_key"] = df["text"].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

    # Relaxed deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["text_key"])  # only drop exact matches
    after = len(df)

    print(f" Removed {before - after} exact duplicate articles")
    return df.drop(columns=["text_key"])

# # === Folder containing CSVs ===
# csv_folder = "../twitter_data"  # adjust as needed
#
# # === Helpers to auto-map column names ===
# def detect_column(columns, options):
#     for col in columns:
#         if any(opt in col.lower() for opt in options):
#             return col
#     return None
#
# # === Process each CSV ===
# for file in os.listdir(csv_folder):
#     if file.endswith(".csv"):
#         symbol = file.replace(".csv", "").upper()
#         file_path = os.path.join(csv_folder, file)
#         print(f"Processing: {file_path}")
#
#         try:
#             df = pd.read_csv(file_path)
#
#             # Identify likely timestamp/text columns
#             timestamp_col = detect_column(df.columns, ["time", "date"])
#             text_col = detect_column(df.columns, ["text", "content", "tweet", "body"])
#
#             if not timestamp_col or not text_col:
#                 raise ValueError("Could not detect 'timestamp' or 'text' columns")
#
#             df = df[[timestamp_col, text_col]].dropna()
#             df = df.rename(columns={timestamp_col: "timestamp", text_col: "text"})
#
#             # Clean and process
#             df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize(None)
#             df = df.dropna(subset=['timestamp'])
#             df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
#             df['source'] = "Twitter"
#             df['symbol'] = symbol
#
#             df = df[['timestamp', 'source', 'text', 'sentiment', 'symbol']]
#             df.to_sql("combined_sentiment_data", con=engine, if_exists="append", index=False)
#             print(f"✅ Inserted {len(df)} rows for {symbol}")
#         except Exception as e:
#             print(f"❌ Failed to process {file}: {e}")

# === Live News from NewsData.io ===
def fetch_live_news_sentiment(symbol):
    print(f" Fetching live news for {symbol} from NewsData.io")
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": symbol,
        "language": "en",
        "category": "business"
    }
    try:
        response = requests.get(url, params=params)
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code}: {response.text}")
        data = []
        for article in response.json().get("results", []):
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if text:
                sentiment = analyzer.polarity_scores(text)["compound"]
                data.append({
                    "timestamp": article.get("pubDate"),
                    "source": "NewsData.io",
                    "text": text,
                    "sentiment": sentiment
                })
        print(f" NewsData.io articles fetched: {len(data)}")
        return data
    except Exception as e:
        print(f"[NewsData Error] {e}")
        return []

# === Historical GNews with API Key Rotation ===
def fetch_gnews_historical_sentiment(symbol, days=30):
    print(f"️ Fetching historical news from GNews API for {symbol} (last {days} days)")
    records = []
    api_index = 0

    for offset in range(days):
        date = datetime.now() - timedelta(days=offset)
        date_str = date.strftime("%Y-%m-%d")

        for attempt in range(len(GNEWS_API_KEYS)):
            token = GNEWS_API_KEYS[api_index % len(GNEWS_API_KEYS)]
            url = "https://gnews.io/api/v4/search"
            params = {
                "q": symbol,
                "from": date_str,
                "to": date_str,
                "lang": "en",
                "token": token,
                "max": 10
            }

            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                print(f"️ {date_str}: {len(articles)} articles")
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}".strip()
                    sentiment = analyzer.polarity_scores(text)["compound"]
                    records.append({
                        "timestamp": article.get("publishedAt"),
                        "source": "GNews.io",
                        "text": text,
                        "sentiment": sentiment
                    })
                break  # success, break inner loop

            elif resp.status_code == 429:
                print(f" [{date_str}] API Key {api_index+1} rate limited. Trying next key after sleep.")
                time.sleep(2)
                api_index += 1
            else:
                print(f" [{date_str}] Error {resp.status_code}. Trying next key.")
                api_index += 1

    print(f" GNews articles fetched: {len(records)}")
    return records

# === Main Execution Block ===
def main():
    for symbol in ticker_list:
        try:
            newsdata_records = fetch_live_news_sentiment(symbol)
            gnews_records = fetch_gnews_historical_sentiment(symbol)

            df_news = pd.DataFrame(newsdata_records)
            df_gnews = pd.DataFrame(gnews_records)

            df_news = clean_df(df_news)
            df_gnews = clean_df(df_gnews)

            df_news["symbol"] = symbol
            df_gnews["symbol"] = symbol

            combined_df = pd.concat([df_news, df_gnews], ignore_index=True).sort_values("timestamp")

            print(f" Cleaned NewsData.io rows: {len(df_news)}")
            print(f" Cleaned GNews.io rows: {len(df_gnews)}")
            print(f" Final Combined Data: {combined_df.shape[0]} rows")

            combined_df.to_sql("combined_sentiment_data", con=engine, if_exists="append", index=False)
            print(f" Saved {symbol} data to SQL\n")
        except Exception as e:
            print(f" Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()
