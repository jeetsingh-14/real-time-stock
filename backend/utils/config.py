"""
Configuration file for the stock sentiment analysis project.
Contains constants used across the project.
"""

# Database connection string
MYSQL_CONNECTION_STRING = "mysql+pymysql://root:285project@127.0.0.1:3307/stock_sentiment"

# List of stock tickers to analyze
TICKERS = ["TSLA", "AAPL", "AMZN", "GOOGL", "GTLB", "AMD", "NVDA", "AAL"]