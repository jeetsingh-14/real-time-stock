import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import io
import re
from PIL import Image

def generate_wordcloud(titles_df):
    """
    Generates a word cloud from news titles.
    
    Args:
        titles_df (pandas.DataFrame): DataFrame containing news titles
        
    Returns:
        PIL.Image: Word cloud image
    """
    if titles_df.empty:
        return None
    
    # Combine all titles into a single string
    text = ' '.join(titles_df['title'].tolist())
    
    # Clean the text
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of', 'as'}
    
    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Convert matplotlib figure to image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Save figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert buffer to image
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

def create_confusion_matrix(confusion_data):
    """
    Creates a confusion matrix visualization using Plotly.
    
    Args:
        confusion_data (dict): Dictionary containing confusion matrix data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not confusion_data:
        return None
    
    # Extract values
    tp = confusion_data.get('true_positive', 0)
    fp = confusion_data.get('false_positive', 0)
    tn = confusion_data.get('true_negative', 0)
    fn = confusion_data.get('false_negative', 0)
    
    # Create confusion matrix
    z = [[tp, fp], [fn, tn]]
    x = ['Predicted Up', 'Predicted Down']
    y = ['Actual Up', 'Actual Down']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=False,
        text=[[f"{tp}", f"{fp}"], [f"{fn}", f"{tn}"]],
        texttemplate="%{text}",
        textfont={"size": 40}
    ))
    
    # Update layout
    fig.update_layout(
        title="Prediction Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_price_chart(df, symbol):
    """
    Creates a price chart with sentiment overlay using Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame containing price and sentiment data
        symbol (str): Stock symbol
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['close_price'],
            name='Close Price',
            line=dict(color='royalblue', width=2)
        )
    )
    
    # Add sentiment as a line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['avg_sentiment'],
            name='Sentiment',
            line=dict(color='firebrick', width=1, dash='dot'),
            yaxis='y2'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price and Sentiment",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="Sentiment Score",
            titlefont=dict(color="firebrick"),
            tickfont=dict(color="firebrick"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[-1, 1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified"
    )
    
    return fig

def create_prediction_accuracy_chart(df, symbol):
    """
    Creates a prediction accuracy chart using Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame containing prediction data
        symbol (str): Stock symbol
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['close_price'],
            name='Actual Price',
            line=dict(color='royalblue', width=2)
        )
    )
    
    # Add markers for correct predictions
    correct_df = df[df['prediction_numeric'] == df['actual_numeric']]
    fig.add_trace(
        go.Scatter(
            x=correct_df['timestamp'],
            y=correct_df['close_price'],
            mode='markers',
            name='Correct Prediction',
            marker=dict(
                color='green',
                size=10,
                symbol='circle',
                line=dict(color='white', width=1)
            )
        )
    )
    
    # Add markers for incorrect predictions
    incorrect_df = df[df['prediction_numeric'] != df['actual_numeric']]
    fig.add_trace(
        go.Scatter(
            x=incorrect_df['timestamp'],
            y=incorrect_df['close_price'],
            mode='markers',
            name='Incorrect Prediction',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                line=dict(color='white', width=1)
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Prediction Accuracy",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified"
    )
    
    return fig

def create_sentiment_by_source_chart(df):
    """
    Creates a bar chart of sentiment by news source using Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sentiment by source data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty:
        return None
    
    # Create color scale based on sentiment
    colors = df['avg_sentiment'].apply(
        lambda x: 'green' if x > 0.2 else ('red' if x < -0.2 else 'gold')
    )
    
    # Create bar chart
    fig = px.bar(
        df,
        x='source',
        y='avg_sentiment',
        color=colors,
        text='article_count',
        labels={'avg_sentiment': 'Average Sentiment', 'source': 'News Source'},
        title='Sentiment by News Source',
        height=400
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=50, r=50, t=50, b=100),
        showlegend=False
    )
    
    # Add count labels
    fig.update_traces(
        texttemplate='%{text} articles',
        textposition='outside'
    )
    
    return fig

def format_prediction_card(prediction_data):
    """
    Formats prediction data for display in a card.
    
    Args:
        prediction_data (dict): Dictionary containing prediction data
        
    Returns:
        tuple: HTML and CSS for the prediction card
    """
    if not prediction_data:
        return None, None
    
    # Determine color based on prediction
    color = "green" if prediction_data['prediction'] == 'up' else "red"
    arrow = "↑" if prediction_data['prediction'] == 'up' else "↓"
    
    # Format timestamp
    timestamp = prediction_data['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(prediction_data['timestamp'], 'strftime') else prediction_data['timestamp']
    
    # Create HTML
    html = f"""
    <div class="prediction-card">
        <h3>Latest Prediction</h3>
        <div class="prediction-value" style="color: {color};">
            {arrow} {prediction_data['prediction'].upper()}
        </div>
        <div class="prediction-details">
            <p>Confidence: <b>{prediction_data['confidence']:.2f}%</b></p>
            <p>Current Price: <b>${prediction_data['current_price']:.2f}</b></p>
            <p>As of: {timestamp}</p>
        </div>
    </div>
    """
    
    # Create CSS
    css = """
    <style>
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .prediction-value {
        font-size: 10rem;
        font-weight: bold;
        margin: 15px 0;
    }
    .prediction-details {
        text-align: left;
    }
    </style>
    """
    
    return html, css

def create_loading_spinner():
    """
    Creates a loading spinner for data operations.
    
    Returns:
        function: A context manager for displaying a loading spinner
    """
    return st.spinner('Loading data...')