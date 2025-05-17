# Real-Time Stock Sentiment Analysis Pipeline

![Stock Sentiment Analysis](https://img.shields.io/badge/Stock-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)

A comprehensive real-time data pipeline for stock sentiment analysis and short-term price movement prediction. This system integrates data ingestion, sentiment analysis, machine learning modeling, and visualization to assist in financial decision-making.

## Project Overview

This project implements an end-to-end pipeline that:

1. Collects and merges sentiment data with real-time stock prices
2. Trains machine learning models to predict price movement direction
3. Generates real-time predictions using sentiment and technical indicators
4. Visualizes results through an interactive Streamlit dashboard
5. Stores all data in a cloud-based data warehouse (Google BigQuery)

## Key Features

- **Sentiment Analysis**: Processes and aggregates sentiment data from multiple sources
- **Price Movement Prediction**: Uses machine learning to predict short-term (3-hour) price movements
- **Real-Time Processing**: Continuously updates predictions as new data becomes available
- **Interactive Dashboard**: Visualizes sentiment trends and predictions through an intuitive interface
- **Cloud Integration**: Stores processed data in Google BigQuery for scalability and accessibility
- **Fallback Mechanisms**: Implements alternative data sources when primary APIs are unavailable

## Project Structure

```
.
├── backend/
│   ├── piplines/
│   │   ├── merge_sentiment_with_price.py   # Merges sentiment and price data
│   │   ├── predict_real_time.py            # Predicts price movements in real time
│   │   ├── save_to_bigquery.py             # Uploads final tables to BigQuery
│   │   └── main.py                         # Pipeline controller
│   ├── training/
│   │   └── train_models.py                 # Trains sentiment-based models
│   └── model/                              # Stores trained models
│       └── best_model.pkl                  # Best performing model
│
├── frontend/
│   └── app.py                              # Streamlit dashboard for visualization
│
├── data/                                   # Data storage directory
│
├── logs/                                   # Log files directory
│   └── pipeline.log                        # Consolidated pipeline logs
│
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # OR
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   # Database connection
   MYSQL_CONNECTION_STRING=mysql+pymysql://username:password@host:port/database

   # API Keys
   TWELVE_DATA_API_KEY=your_twelve_data_api_key

   # BigQuery credentials
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   ```

## Configuration

The project requires:

1. **MySQL Database**: Set up with the following tables:
   - `combined_sentiment_data`: Historical sentiment scores from multiple sources
   - `merged_price_sentiment`: Merged data used for model training
   - `real_time_predictions`: Real-time model outputs (UP, DOWN, NEUTRAL)

2. **API Access**:
   - yFinance (free, no API key required)
   - Twelve Data API (as fallback for stock prices)

3. **Google BigQuery**: For cloud-based data warehousing

## Usage

### Running the Full Pipeline

Execute the complete pipeline with:

```bash
python backend\main.py
```

This will:
1. Merge sentiment data with stock prices
2. Train prediction models
3. Generate real-time predictions
4. Upload results to BigQuery

### Launching the Dashboard

Start the interactive dashboard with:

```bash
streamlit run frontend\app.py
```

The dashboard provides:
- Sentiment trend analysis
- Latest model predictions
- Manual prediction testing

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
   - Uploads all three key tables to BigQuery

5. **Dashboard**
   - Streamlit app with three tabs:
     - Sentiment trend analysis
     - Latest model predictions
     - Manual prediction testing

## Technologies Used

- **Python 3.10**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, joblib
- **Data Sources**: yFinance, Twelve Data API
- **Database**: MySQL, SQLAlchemy
- **Cloud Storage**: Google BigQuery
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Utilities**: python-dateutil, pathlib

## Important Notes

- Ensure the MySQL server is running and the schema matches expected table structures
- Respect API rate limits, especially for yFinance and Twelve Data
- Best model and logs are stored under the `model/` and `logs/` directories respectively
- The dashboard requires an active database connection to function properly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Jeet Singh Saini (MSBA 2025)

## Acknowledgements

- This project was developed as part of the MSBA 285 Final Capstone Project
- Special thanks to all contributors and data providers
