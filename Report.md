# Real-Time Stock Sentiment Analysis Pipeline

This project implements a real-time data pipeline for stock sentiment analysis and short-term price movement prediction. Developed as part of the MSBA 285 Final Capstone Project, the system integrates data ingestion, modeling, and visualization to assist in financial decision-making.

---

## Project Objectives

- Merge sentiment data with real-time stock prices
- Train supervised machine learning models to predict price movement direction
- Generate real-time predictions using technical indicators
- Deploy a live dashboard using Streamlit
- Store all results in a cloud-based data warehouse (Google BigQuery)

---

## Project Structure

```bash
.
├── backend/
│   ├── merge_sentiment_with_price.py      # Step 1: Merge sentiment and price data
│   ├── train_models.py                    # Step 2: Train sentiment-based models
│   ├── predict_real_time.py               # Step 3: Predict price movements in real time
│   ├── save_to_bigquery.py                # Step 4: Upload final tables to BigQuery
│   ├── train_tech_model.py                # Train backup model using technical indicators only
│   └── model/                             # Stores trained models (best_model.pkl, tech_model.pkl)

├── frontend/
│   └── app.py                             # Streamlit dashboard for visualizing results

├── logs/
│   └── pipeline.log                       # Consolidated pipeline logs

├── main.py                                # Pipeline controller to run all stages sequentially

└── requirements.txt                       # Python dependencies
```

---

## Technologies Used

- Python 3.10
- pandas, scikit-learn, xgboost, SQLAlchemy, joblib
- yFinance and Twelve Data API for stock prices
- MySQL for relational storage
- Google BigQuery for cloud-based warehousing
- Streamlit and Plotly for visualization

---

## Pipeline Workflow

1. **Data Merge**
   - Loads 30 days of sentiment scores from MySQL
   - Downloads hourly price data (yFinance or fallback to Twelve Data)
   - Joins sentiment and price data on date

2. **Model Training**
   - Creates features from sentiment and timestamps
   - Trains Logistic Regression, Random Forest, and XGBoost
   - Selects and saves the best-performing model

3. **Real-Time Prediction**
   - Extracts technical indicators from latest stock data
   - Predicts short-term movement (3-hour horizon)
   - Stores outputs to `real_time_predictions` table

4. **BigQuery Upload**
   - Uploads all three key tables to BigQuery:
     - `real_time_predictions`
     - `merged_price_sentiment`
     - `combined_sentiment_data`

5. **Dashboard**
   - Streamlit app with three tabs:
     - Sentiment trend analysis
     - Latest model predictions
     - Manual prediction testing

---

## Key Tables

| Table Name               | Description                                 |
|--------------------------|---------------------------------------------|
| `combined_sentiment_data`| Historical sentiment scores from multiple sources |
| `merged_price_sentiment` | Merged data used for model training         |
| `real_time_predictions`  | Real-time model outputs (UP, DOWN, NEUTRAL) |

---

## How to Run

1. Set up the `.env` file or environment variables for:
   - MySQL connection string
   - Twelve Data API Key
   - BigQuery credentials

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute the full pipeline:
   ```bash
   python main.py
   ```

4. Launch the dashboard:
   ```bash
   streamlit run frontend/app.py
   ```

---

## Authors

- Jeet Singh Saini (MSBA 2025)

---

## Notes

- Make sure the MySQL server is running and the schema matches expected table structures.
- Ensure API rate limits are respected, especially for yFinance and Twelve Data.
- Best model and logs are stored under the `model/` and `logs/` directories respectively.