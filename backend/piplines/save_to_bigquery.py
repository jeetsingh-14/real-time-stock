import pandas as pd
from sqlalchemy import create_engine
from pandas_gbq import to_gbq
import logging
import os
from pathlib import Path
from stockmarket_analysis.backend.utils.config import MYSQL_CONNECTION_STRING

# === Configuration ===
db_url = MYSQL_CONNECTION_STRING

bq_project_id = "msba285-spring25-student"
# Mapping of table names to BigQuery destinations
tables_to_upload = {
    "real_time_predictions": "stock_predictions.real_time_predictions",
    "merged_price_sentiment": "stock_predictions.merged_price_sentiment",
    "combined_sentiment_data": "stock_predictions.combined_sentiment_data"
}

# Setup Logging
log_dir = os.path.join(Path(__file__).resolve().parents[2], "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "bigquery_upload.log")

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def export_table_to_bigquery(local_table, bq_table):
    try:
        logger.info(f"Connecting to MySQL and reading from '{local_table}'")
        engine = create_engine(db_url)

        query = f"SELECT * FROM {local_table}"
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning(f"No records found in '{local_table}' table. Skipping upload.")
            return

        # Format timestamp to 'YYYY-MM-DD HH:MM:SS' string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Uploading {len(df)} records to BigQuery table: {bq_table}")

        to_gbq(
            dataframe=df,
            destination_table=bq_table,
            project_id=bq_project_id,
            if_exists="replace",
            # progress_bar_type=None
        )

        logger.info(f"Successfully uploaded to {bq_table}")
    except Exception as e:
        logger.error(f"Error exporting '{local_table}' to BigQuery: {str(e)}")

def main():
    for local_table, bq_table in tables_to_upload.items():
        export_table_to_bigquery(local_table, bq_table)

if __name__ == "__main__":
    main()
