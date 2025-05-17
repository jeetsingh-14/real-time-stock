import mysql.connector
import pandas as pd
import streamlit as st

@st.cache_resource
def get_database_connection():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            port=3307,
            user="root",
            password="285project",
            database="stock_sentiment"
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_symbols():
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        query = "SELECT DISTINCT symbol FROM combined_sentiment_data ORDER BY symbol"
        df = pd.read_sql(query, conn)
        return df['symbol'].tolist()
    except Exception as e:
        st.error(f"Error retrieving symbols: {e}")
        return []
    finally:
        conn.close()

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_latest_prediction(symbol):
    """
    Retrieves the latest prediction for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get predictions for
        
    Returns:
        dict: A dictionary containing the latest prediction data
    """
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT p.*, s.close_price, s.timestamp 
        FROM predictions p
        JOIN stock_data s ON p.symbol = s.symbol AND p.prediction_for = s.timestamp
        WHERE p.symbol = %s
        ORDER BY p.prediction_for DESC
        LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(symbol,))
        
        if df.empty:
            return None
            
        return {
            'symbol': symbol,
            'prediction': df.iloc[0]['prediction'],
            'confidence': df.iloc[0]['confidence'],
            'timestamp': df.iloc[0]['timestamp'],
            'current_price': df.iloc[0]['close_price']
        }
    except Exception as e:
        st.error(f"Error retrieving latest prediction: {e}")
        return None
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_historical_predictions(symbol, days=30):
    """
    Retrieves historical predictions for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get predictions for
        days (int): Number of days of historical data to retrieve
        
    Returns:
        pandas.DataFrame: A DataFrame containing historical prediction data
    """
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT p.*, s.close_price, s.open_price, s.high_price, s.low_price, s.volume, s.timestamp 
        FROM predictions p
        JOIN stock_data s ON p.symbol = s.symbol AND p.prediction_for = s.timestamp
        WHERE p.symbol = %s AND s.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
        ORDER BY s.timestamp
        """
        df = pd.read_sql(query, conn, params=(symbol, days))
        
        if not df.empty:
            # Convert prediction to numeric for easier comparison
            df['prediction_numeric'] = df['prediction'].apply(lambda x: 1 if x == 'up' else 0)
            # Calculate actual movement
            df['actual_movement'] = df['close_price'].diff().shift(-1).apply(lambda x: 'up' if x > 0 else 'down')
            df['actual_numeric'] = df['actual_movement'].apply(lambda x: 1 if x == 'up' else 0)
            
        return df
    except Exception as e:
        st.error(f"Error retrieving historical predictions: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_historical_price_sentiment(symbol, days=30):
    """
    Retrieves historical price and sentiment data for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get data for
        days (int): Number of days of historical data to retrieve
        
    Returns:
        pandas.DataFrame: A DataFrame containing historical price and sentiment data
    """
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT s.timestamp, s.close_price, s.open_price, s.high_price, s.low_price, 
               s.volume, COALESCE(AVG(n.sentiment_score), 0) as avg_sentiment
        FROM stock_data s
        LEFT JOIN news n ON s.symbol = n.symbol AND DATE(s.timestamp) = DATE(n.published_at)
        WHERE s.symbol = %s AND s.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
        GROUP BY s.timestamp, s.close_price, s.open_price, s.high_price, s.low_price, s.volume
        ORDER BY s.timestamp
        """
        df = pd.read_sql(query, conn, params=(symbol, days))
        return df
    except Exception as e:
        st.error(f"Error retrieving historical price and sentiment: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_latest_news(symbol, limit=10):
    """
    Retrieves the latest news for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get news for
        limit (int): Maximum number of news items to retrieve
        
    Returns:
        pandas.DataFrame: A DataFrame containing the latest news
    """
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT title, source, url, published_at, sentiment_score
        FROM news
        WHERE symbol = %s
        ORDER BY published_at DESC
        LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(symbol, limit))
        return df
    except Exception as e:
        st.error(f"Error retrieving latest news: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_sentiment_by_source(symbol, days=30):
    """
    Retrieves sentiment data grouped by news source for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get sentiment data for
        days (int): Number of days of historical data to retrieve
        
    Returns:
        pandas.DataFrame: A DataFrame containing sentiment data grouped by source
    """
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT source, AVG(sentiment_score) as avg_sentiment, COUNT(*) as article_count
        FROM news
        WHERE symbol = %s AND published_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        GROUP BY source
        ORDER BY avg_sentiment DESC
        """
        df = pd.read_sql(query, conn, params=(symbol, days))
        return df
    except Exception as e:
        st.error(f"Error retrieving sentiment by source: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_news_titles_for_wordcloud(symbol, days=30):
    """
    Retrieves news titles for generating a word cloud for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to get news titles for
        days (int): Number of days of historical data to retrieve
        
    Returns:
        pandas.DataFrame: A DataFrame containing news titles
    """
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT title
        FROM news
        WHERE symbol = %s AND published_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        df = pd.read_sql(query, conn, params=(symbol, days))
        return df
    except Exception as e:
        st.error(f"Error retrieving news titles: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_prediction_accuracy(symbol, days=30):
    """
    Calculates prediction accuracy for a given stock symbol.
    
    Args:
        symbol (str): The stock symbol to calculate prediction accuracy for
        days (int): Number of days of historical data to use
        
    Returns:
        tuple: A tuple containing (accuracy_percentage, confusion_matrix_data)
    """
    df = get_historical_predictions(symbol, days)
    
    if df.empty:
        return 0, pd.DataFrame()
    
    # Remove rows with NaN values in relevant columns
    df = df.dropna(subset=['prediction_numeric', 'actual_numeric'])
    
    if df.empty:
        return 0, pd.DataFrame()
    
    # Calculate accuracy
    correct_predictions = (df['prediction_numeric'] == df['actual_numeric']).sum()
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # Create confusion matrix data
    confusion_data = {
        'true_positive': ((df['prediction_numeric'] == 1) & (df['actual_numeric'] == 1)).sum(),
        'false_positive': ((df['prediction_numeric'] == 1) & (df['actual_numeric'] == 0)).sum(),
        'true_negative': ((df['prediction_numeric'] == 0) & (df['actual_numeric'] == 0)).sum(),
        'false_negative': ((df['prediction_numeric'] == 0) & (df['actual_numeric'] == 1)).sum()
    }
    
    return accuracy, confusion_data