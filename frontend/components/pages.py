import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta

# Import local modules
from stockmarket_analysis.frontend.components.database import (
    get_latest_prediction, get_historical_predictions, get_historical_price_sentiment,
    get_latest_news, get_sentiment_by_source, get_news_titles_for_wordcloud,
    get_prediction_accuracy
)
from stockmarket_analysis.frontend.components.utils import (
    generate_wordcloud, create_confusion_matrix, create_price_chart,
    create_prediction_accuracy_chart, create_sentiment_by_source_chart,
    format_prediction_card, create_loading_spinner
)
from stockmarket_analysis.frontend.components.grafana_embed import (
    create_grafana_dashboard, get_sample_panels, grafana_settings_ui
)

def landing_page():
    """
    Creates a landing page for the dashboard.
    """
    # Logo and title
    col1, col2 = st.columns([1, 3])

    with col1:
        # Placeholder for logo
        st.image("https://via.placeholder.com/150x150?text=LOGO", width=150)

    with col2:
        st.title("Stock Market Prediction Dashboard")
        st.markdown("""
        Welcome to the Stock Market Prediction Dashboard. This dashboard provides real-time 
        stock market predictions, sentiment analysis, and historical data visualization.

        Use the navigation menu on the left to explore different sections of the dashboard.
        """)

    # Dashboard sections
    st.subheader("Dashboard Sections")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Overview
        Get a quick overview of the latest predictions and market sentiment.

        ### Trends
        Visualize historical price and sentiment trends.
        """)

    with col2:
        st.markdown("""
        ### Prediction Metrics
        Analyze prediction accuracy and performance metrics.

        ### Live News
        View the latest news and sentiment analysis.
        """)

    # Grafana integration
    st.subheader("Grafana Integration")
    st.markdown("""
    This dashboard integrates with Grafana to provide additional visualizations and metrics.
    Visit the Grafana Panels section to view embedded Grafana dashboards.
    """)

    # Last updated
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

def overview_page(symbol):
    """
    Creates an overview page with the latest prediction and key metrics.

    Args:
        symbol (str): Stock symbol to display data for
    """
    st.header(f"Overview: {symbol}")

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Latest prediction card
        with create_loading_spinner():
            prediction_data = get_latest_prediction(symbol)

        if prediction_data:
            html, css = format_prediction_card(prediction_data)
            if html and css:
                st.markdown(css, unsafe_allow_html=True)
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.error(f"No prediction data available for {symbol}")

        # Prediction accuracy
        st.subheader("Prediction Accuracy")
        with create_loading_spinner():
            accuracy, confusion_data = get_prediction_accuracy(symbol)

        if accuracy > 0:
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")

            # Confusion matrix in expander
            with st.expander("View Confusion Matrix"):
                confusion_fig = create_confusion_matrix(confusion_data)
                if confusion_fig:
                    st.plotly_chart(confusion_fig, use_container_width=True)
        else:
            st.info(f"No accuracy data available for {symbol}")

    with col2:
        # Price and sentiment chart
        st.subheader("Recent Price and Sentiment")
        with create_loading_spinner():
            price_sentiment_df = get_historical_price_sentiment(symbol, days=14)

        if not price_sentiment_df.empty:
            fig = create_price_chart(price_sentiment_df, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No recent price and sentiment data available for {symbol}")

        # Latest news
        st.subheader("Latest News")
        with create_loading_spinner():
            news_df = get_latest_news(symbol, limit=5)

        if not news_df.empty:
            for _, row in news_df.iterrows():
                sentiment_color = "green" if row['sentiment_score'] > 0.2 else ("red" if row['sentiment_score'] < -0.2 else "orange")
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid {sentiment_color};">
                    <h4 style="margin: 0;">{row['title']}</h4>
                    <p style="margin: 5px 0; color: gray;">
                        {row['source']} | {row['published_at'].strftime('%Y-%m-%d %H:%M')}
                    </p>
                    <p>Sentiment: <span style="color: {sentiment_color}; font-weight: bold;">
                        {row['sentiment_score']:.2f}
                    </span></p>
                    <a href="{row['url']}" target="_blank">Read more</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No recent news available for {symbol}")

def trends_page(symbol, days):
    """
    Creates a trends page with historical price, sentiment, and prediction data.

    Args:
        symbol (str): Stock symbol to display data for
        days (int): Number of days of historical data to display
    """
    st.header(f"Trends: {symbol}")

    # Historical price and sentiment
    with st.expander("Price and Sentiment Trends", expanded=True):
        st.subheader("Historical Price and Sentiment")
        with create_loading_spinner():
            price_sentiment_df = get_historical_price_sentiment(symbol, days=days)

        if not price_sentiment_df.empty:
            fig = create_price_chart(price_sentiment_df, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No historical price and sentiment data available for {symbol}")

    # Prediction accuracy
    with st.expander("Prediction Accuracy Trends", expanded=True):
        st.subheader("Historical Prediction Accuracy")
        with create_loading_spinner():
            predictions_df = get_historical_predictions(symbol, days=days)

        if not predictions_df.empty:
            fig = create_prediction_accuracy_chart(predictions_df, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Calculate rolling accuracy
            if len(predictions_df) > 5:
                predictions_df['correct'] = predictions_df['prediction_numeric'] == predictions_df['actual_numeric']
                predictions_df['rolling_accuracy'] = predictions_df['correct'].rolling(window=5).mean() * 100

                # Plot rolling accuracy
                import plotly.express as px
                fig = px.line(
                    predictions_df.dropna(subset=['rolling_accuracy']), 
                    x='timestamp', 
                    y='rolling_accuracy',
                    title=f"{symbol} 5-Day Rolling Prediction Accuracy",
                    labels={'rolling_accuracy': 'Accuracy (%)', 'timestamp': 'Date'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No historical prediction data available for {symbol}")

    # Sentiment by source
    with st.expander("Sentiment by News Source", expanded=True):
        st.subheader("Sentiment Analysis by News Source")
        with create_loading_spinner():
            sentiment_source_df = get_sentiment_by_source(symbol, days=days)

        if not sentiment_source_df.empty:
            fig = create_sentiment_by_source_chart(sentiment_source_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No sentiment by source data available for {symbol}")

    # Word cloud
    with st.expander("News Word Cloud", expanded=True):
        st.subheader("News Titles Word Cloud")
        with create_loading_spinner():
            titles_df = get_news_titles_for_wordcloud(symbol, days=days)

        if not titles_df.empty:
            wordcloud_img = generate_wordcloud(titles_df)
            if wordcloud_img:
                st.image(wordcloud_img, use_column_width=True)
        else:
            st.info(f"No news titles available for {symbol}")

def prediction_metrics_page(symbol, days):
    """
    Creates a prediction metrics page with detailed accuracy analysis.

    Args:
        symbol (str): Stock symbol to display data for
        days (int): Number of days of historical data to display
    """
    st.header(f"Prediction Metrics: {symbol}")

    # Get prediction data
    with create_loading_spinner():
        predictions_df = get_historical_predictions(symbol, days=days)
        accuracy, confusion_data = get_prediction_accuracy(symbol, days=days)

    if predictions_df.empty:
        st.info(f"No prediction data available for {symbol}")
        return

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1f}%")

    with col2:
        if 'prediction' in predictions_df.columns:
            up_accuracy = predictions_df[predictions_df['prediction'] == 'up']
            if not up_accuracy.empty:
                up_correct = (up_accuracy['prediction'] == up_accuracy['actual_movement']).mean() * 100
                st.metric("'Up' Prediction Accuracy", f"{up_correct:.1f}%")
            else:
                st.metric("'Up' Prediction Accuracy", "N/A")

    with col3:
        if 'prediction' in predictions_df.columns:
            down_accuracy = predictions_df[predictions_df['prediction'] == 'down']
            if not down_accuracy.empty:
                down_correct = (down_accuracy['prediction'] == down_accuracy['actual_movement']).mean() * 100
                st.metric("'Down' Prediction Accuracy", f"{down_correct:.1f}%")
            else:
                st.metric("'Down' Prediction Accuracy", "N/A")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    confusion_fig = create_confusion_matrix(confusion_data)
    if confusion_fig:
        st.plotly_chart(confusion_fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("Detailed Prediction Metrics")

    if not predictions_df.empty:
        # Calculate additional metrics
        tp = confusion_data.get('true_positive', 0)
        fp = confusion_data.get('false_positive', 0)
        tn = confusion_data.get('true_negative', 0)
        fn = confusion_data.get('false_negative', 0)

        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy / 100, precision, recall, f1],
            'Description': [
                'Percentage of correct predictions',
                'Percentage of correct "up" predictions out of all "up" predictions',
                'Percentage of correct "up" predictions out of all actual "up" movements',
                'Harmonic mean of precision and recall'
            ]
        })

        # Format the Value column as percentage
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2%}")

        st.dataframe(metrics_df, use_container_width=True)

        # Prediction history table
        st.subheader("Recent Prediction History")

        # Prepare a clean dataframe for display
        display_df = predictions_df.copy()
        display_df['correct'] = display_df['prediction'] == display_df['actual_movement']
        display_df = display_df[['timestamp', 'prediction', 'confidence', 'actual_movement', 'correct', 'close_price']]
        display_df = display_df.rename(columns={
            'timestamp': 'Date',
            'prediction': 'Prediction',
            'confidence': 'Confidence',
            'actual_movement': 'Actual',
            'correct': 'Correct',
            'close_price': 'Close Price'
        })

        # Format the dataframe
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}%")
        display_df['Correct'] = display_df['Correct'].apply(lambda x: '✅' if x else '❌')

        st.dataframe(display_df.sort_values('Date', ascending=False).head(10), use_container_width=True)
    else:
        st.info(f"No prediction metrics available for {symbol}")

def live_news_page(symbol, limit=20):
    """
    Creates a live news page with the latest news and sentiment analysis.

    Args:
        symbol (str): Stock symbol to display news for
        limit (int): Maximum number of news items to display
    """
    st.header(f"Live News: {symbol}")

    # Get latest news
    with create_loading_spinner():
        news_df = get_latest_news(symbol, limit=limit)

    if news_df.empty:
        st.info(f"No news available for {symbol}")
        return

    # News filters
    col1, col2 = st.columns(2)

    with col1:
        # Filter by source
        sources = ['All'] + sorted(news_df['source'].unique().tolist())
        selected_source = st.selectbox("Filter by Source", sources)

    with col2:
        # Filter by sentiment
        sentiment_options = ['All', 'Positive', 'Neutral', 'Negative']
        selected_sentiment = st.selectbox("Filter by Sentiment", sentiment_options)

    # Apply filters
    filtered_df = news_df.copy()

    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]

    if selected_sentiment != 'All':
        if selected_sentiment == 'Positive':
            filtered_df = filtered_df[filtered_df['sentiment_score'] > 0.2]
        elif selected_sentiment == 'Negative':
            filtered_df = filtered_df[filtered_df['sentiment_score'] < -0.2]
        else:  # Neutral
            filtered_df = filtered_df[(filtered_df['sentiment_score'] >= -0.2) & (filtered_df['sentiment_score'] <= 0.2)]

    # Display news
    if filtered_df.empty:
        st.info("No news matching the selected filters")
    else:
        # Sentiment distribution
        st.subheader("Sentiment Distribution")

        # Create sentiment categories
        filtered_df['sentiment_category'] = filtered_df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )

        # Count by category
        sentiment_counts = filtered_df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Create pie chart
        import plotly.express as px
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': 'green', 'Neutral': 'gold', 'Negative': 'red'},
            title="News Sentiment Distribution"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # News list
        st.subheader(f"Latest News ({len(filtered_df)} articles)")

        for _, row in filtered_df.iterrows():
            sentiment_color = "green" if row['sentiment_score'] > 0.2 else ("red" if row['sentiment_score'] < -0.2 else "orange")
            sentiment_label = "Positive" if row['sentiment_score'] > 0.2 else ("Negative" if row['sentiment_score'] < -0.2 else "Neutral")

            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-left: 5px solid {sentiment_color};">
                <h3 style="margin: 0;">{row['title']}</h3>
                <p style="margin: 5px 0; color: gray;">
                    {row['source']} | {row['published_at'].strftime('%Y-%m-%d %H:%M')}
                </p>
                <p>Sentiment: <span style="color: {sentiment_color}; font-weight: bold;">
                    {sentiment_label} ({row['sentiment_score']:.2f})
                </span></p>
                <a href="{row['url']}" target="_blank" style="text-decoration: none;">
                    <button style="background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer;">
                        Read Article
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)

def grafana_panels_page():
    """
    Creates a page for displaying Grafana panels.
    """
    st.header("Grafana Panels")

    # Get panels from session state or initialize
    if 'grafana_panels' not in st.session_state:
        st.session_state.grafana_panels = get_sample_panels()

    # Create dashboard
    create_grafana_dashboard(st.session_state.grafana_panels)

def settings_page():
    """
    Creates a settings page for configuring dashboard options.
    """
    st.header("Dashboard Settings")

    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["General Settings", "Grafana Integration", "Data Settings"])

    with tab1:
        st.subheader("General Settings")

        # Theme settings
        st.write("Theme Settings")
        theme = st.selectbox("Dashboard Theme", ["Light", "Dark"], index=0)

        # Auto-refresh settings
        st.write("Auto-Refresh Settings")
        enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)

        if enable_auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)

            # Save settings to session state
            if st.button("Save General Settings"):
                st.session_state.theme = theme
                st.session_state.enable_auto_refresh = enable_auto_refresh
                st.session_state.refresh_interval = refresh_interval
                st.success("General settings saved!")

    with tab2:
        # Grafana settings
        grafana_settings_ui()

    with tab3:
        st.subheader("Data Settings")

        # Default symbol
        st.write("Default Symbol")
        default_symbol = st.text_input("Default Stock Symbol", "AAPL")

        # Default time range
        st.write("Default Time Range")
        default_days = st.slider("Default Days of Historical Data", 7, 90, 30)

        # Cache settings
        st.write("Cache Settings")
        cache_ttl = st.slider("Cache Time-to-Live (minutes)", 1, 60, 5)

        # Save settings to session state
        if st.button("Save Data Settings"):
            st.session_state.default_symbol = default_symbol
            st.session_state.default_days = default_days
            st.session_state.cache_ttl = cache_ttl
            st.success("Data settings saved!")
