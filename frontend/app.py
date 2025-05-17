import streamlit as st

# Set page config (must be the first Streamlit call)
st.set_page_config(
    page_title="Stock Sentiment Analysis Dashboard",
    page_icon="",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import joblib
import os
import sys
from pathlib import Path

# === Configuration ===
db_url = "mysql+pymysql://root:285project@127.0.0.1:3307/stock_sentiment"
model_path = os.path.join(
    Path(__file__).resolve().parents[1],  # stockmarket_analysis/
    "backend", "model", "best_model.pkl"
)
ticker_list = ["TSLA", "AAPL", "AMZN", "GOOGL", "GTLB", "AMD", "NVDA", "AAL"]

# === Database connection ===
@st.cache_resource
def get_db_connection():
    return create_engine(db_url)

engine = get_db_connection()

# === Load the trained model ===
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found. Run train_models.py first.")
        return None
    return joblib.load(model_path)

model = load_model()

# === Helper functions ===
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_sentiment_data(symbol, days=7):
    query = f"""
    SELECT timestamp, sentiment, source
    FROM combined_sentiment_data
    WHERE symbol = '{symbol}'
    AND timestamp >= NOW() - INTERVAL {days} DAY
    ORDER BY timestamp ASC
    """

    try:
        df = pd.read_sql(query, con=engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error fetching sentiment data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_predictions():
    query = """
    SELECT r.*
    FROM real_time_predictions r
    INNER JOIN (
        SELECT symbol, MAX(timestamp) as max_timestamp
        FROM real_time_predictions
        WHERE prediction != 'ERROR'
        GROUP BY symbol
    ) latest ON r.symbol = latest.symbol AND r.timestamp = latest.max_timestamp
    WHERE r.prediction != 'ERROR'
    ORDER BY r.symbol;
    """

    try:
        df = pd.read_sql(query, con=engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return pd.DataFrame()

def predict_custom_input(sentiment, price):
    if model is None:
        return "Model not loaded"

    try:
        # Get current time features
        now = datetime.now()
        day_of_week = now.weekday()
        hour = now.hour
        month = now.month

        # Create a feature array with the same structure as training data
        features = np.array([[sentiment, day_of_week, hour, month]])

        # Make prediction - access the model from the dictionary
        prediction_code = model['model'].predict(features)[0]

        # Convert numeric prediction back to label
        label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        prediction = label_map.get(prediction_code, "UNKNOWN")

        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return "ERROR"

# === App UI ===
st.title(" Stock Sentiment Analysis Dashboard")
st.markdown("Real-time sentiment analysis and price movement prediction")

# === Sidebar ===
st.sidebar.header("Settings")
selected_ticker = st.sidebar.selectbox("Select Stock", ticker_list)
days_to_show = st.sidebar.slider("Days of Sentiment History", 1, 30, 7)

# === Main Content ===
# Create tabs
tab1, tab2, tab3 = st.tabs([" Live Sentiment", " Predictions", " Test Model"])

# === Tab 1: Live Sentiment ===
with tab1:
    st.header(f"Sentiment Analysis for {selected_ticker}")

    # Fetch sentiment data
    sentiment_df = fetch_sentiment_data(selected_ticker, days_to_show)

    if not sentiment_df.empty:
        # Calculate daily average sentiment
        daily_sentiment = sentiment_df.copy()
        daily_sentiment['date'] = daily_sentiment['timestamp'].dt.date
        daily_avg = daily_sentiment.groupby('date')['sentiment'].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])

        # Plot sentiment over time
        st.subheader("Sentiment Trend")
        fig = px.line(
            daily_avg,
            x="date",
            y="sentiment",
            title=f"{selected_ticker} Daily Average Sentiment",
            labels={"sentiment": "Sentiment Score", "date": "Date"}
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode="x unified"
        )
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=daily_avg['date'].min(),
            y0=0,
            x1=daily_avg['date'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Source breakdown
        st.subheader("Sentiment by Source")
        source_counts = sentiment_df['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                source_counts,
                values='Count',
                names='Source',
                title="Sentiment Sources"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            source_sentiment = sentiment_df.groupby('source')['sentiment'].mean().reset_index()
            source_sentiment.columns = ['Source', 'Average Sentiment']

            fig = px.bar(
                source_sentiment,
                x='Source',
                y='Average Sentiment',
                title="Average Sentiment by Source",
                color='Average Sentiment',
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("View Raw Sentiment Data"):
            st.dataframe(sentiment_df)
    else:
        st.warning(f"No sentiment data available for {selected_ticker} in the last {days_to_show} days")

# === Tab 2: Predictions ===
with tab2:
    st.header("Latest Price Movement Predictions")

    # Fetch latest predictions
    predictions_df = fetch_latest_predictions()

    if not predictions_df.empty:
        # Create a dashboard of predictions
        cols = st.columns(len(predictions_df))

        for i, (_, row) in enumerate(predictions_df.iterrows()):
            with cols[i]:
                symbol = row['symbol']
                prediction = row['prediction']
                price = row['current_price']
                sentiment = row['avg_sentiment']
                timestamp = row['timestamp'].strftime("%Y-%m-%d %H:%M")

                # Set color based on prediction
                if prediction == "UP":
                    color = "green"
                    icon = "⬆️"
                elif prediction == "DOWN":
                    color = "red"
                    icon = "⬇️"
                else:
                    color = "gray"
                    icon = "️➖"

                st.markdown(f"### {symbol} {icon}")
                st.markdown(f"**Prediction:** <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
                st.markdown(f"**Price:** ${price:.2f}")
                st.markdown(f"**Sentiment:** {f'{sentiment:.4f}' if sentiment is not None else 'N/A'}")
                st.markdown(f"**As of:** {timestamp}")

        # Show all predictions in a table
        with st.expander("View All Predictions"):
            st.dataframe(predictions_df)
    else:
        st.warning("No predictions available. Run predict_real_time.py to generate predictions.")

# === Tab 3: Test Model ===
with tab3:
    st.header("Test the Prediction Model")
    st.markdown("Enter custom sentiment and price values to see what the model would predict")

    col1, col2 = st.columns(2)

    with col1:
        custom_sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
        custom_price = st.number_input("Stock Price ($)", min_value=1.0, max_value=10000.0, value=100.0)

        if st.button("Predict"):
            prediction = predict_custom_input(custom_sentiment, custom_price)

            # Display prediction with appropriate styling
            if prediction == "UP":
                st.markdown("### Prediction: <span style='color:green'>UP </span>", unsafe_allow_html=True)
            elif prediction == "DOWN":
                st.markdown("### Prediction: <span style='color:red'>DOWN </span>", unsafe_allow_html=True)
            elif prediction == "NEUTRAL":
                st.markdown("### Prediction: <span style='color:gray'>NEUTRAL ️</span>", unsafe_allow_html=True)
            else:
                st.error(f"Error: {prediction}")

    with col2:
        st.markdown("### How to interpret sentiment scores:")
        st.markdown("""
        - **-1.0 to -0.5**: Very negative sentiment
        - **-0.5 to -0.1**: Negative sentiment
        - **-0.1 to 0.1**: Neutral sentiment
        - **0.1 to 0.5**: Positive sentiment
        - **0.5 to 1.0**: Very positive sentiment
        """)

        st.markdown("### About the model:")
        if model is not None:
            model_type = type(model).__name__
            st.markdown(f"- **Model type:** {model_type}")

            # Try to get feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.markdown("### Feature Importance:")
                importances = model.feature_importances_
                features = ["Sentiment", "Price"]

                fig = px.bar(
                    x=features,
                    y=importances,
                    labels={'x': 'Feature', 'y': 'Importance'},
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, 'coef_'):
                st.markdown("### Feature Coefficients:")
                coefficients = model.coef_[0]
                features = ["Sentiment", "Price"]

                fig = px.bar(
                    x=features,
                    y=coefficients,
                    labels={'x': 'Feature', 'y': 'Coefficient'},
                    title="Feature Coefficients"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model not loaded. Run train_models.py first.")
