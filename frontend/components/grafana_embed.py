import streamlit as st
import pandas as pd

def embed_grafana_panel(panel_url, height=450):
    """
    Embeds a Grafana panel in the Streamlit dashboard using an iframe.
    
    Args:
        panel_url (str): URL of the Grafana panel to embed
        height (int): Height of the iframe in pixels
        
    Returns:
        None: Displays the iframe directly
    """
    # Add theme parameter to match Streamlit theme
    if "?" in panel_url:
        panel_url += "&theme=light"
    else:
        panel_url += "?theme=light"
    
    # Create iframe HTML
    iframe_html = f"""
    <iframe
        src="{panel_url}"
        width="100%"
        height="{height}"
        frameborder="0"
        style="border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"
    ></iframe>
    """
    
    # Display iframe
    st.markdown(iframe_html, unsafe_allow_html=True)

def create_grafana_dashboard(panels):
    """
    Creates a dashboard with multiple Grafana panels.
    
    Args:
        panels (list): List of dictionaries containing panel information
                       Each dictionary should have 'title', 'url', and optionally 'height'
        
    Returns:
        None: Displays the panels directly
    """
    if not panels:
        st.info("No Grafana panels configured. Please add panel URLs in the settings.")
        return
    
    # Display each panel with a title
    for panel in panels:
        st.subheader(panel['title'])
        embed_grafana_panel(panel['url'], panel.get('height', 450))
        st.markdown("---")

def get_sample_panels():
    """
    Returns a list of sample Grafana panels for demonstration.
    In a real application, these would come from configuration or user input.
    
    Returns:
        list: List of dictionaries containing panel information
    """
    return [
        {
            'title': 'Stock Price Overview',
            'url': 'http://localhost:3000/d/stock-overview?orgId=1',
            'height': 400
        },
        {
            'title': 'Market Sentiment Analysis',
            'url': 'http://localhost:3000/d/sentiment-analysis?orgId=1',
            'height': 350
        },
        {
            'title': 'Trading Volume Metrics',
            'url': 'http://localhost:3000/d/volume-metrics?orgId=1',
            'height': 300
        }
    ]

def grafana_settings_ui():
    """
    Creates a UI for configuring Grafana panel settings.
    
    Returns:
        list: List of configured panel dictionaries
    """
    st.subheader("Grafana Integration Settings")
    
    # Get existing panels from session state or initialize
    if 'grafana_panels' not in st.session_state:
        st.session_state.grafana_panels = get_sample_panels()
    
    panels = st.session_state.grafana_panels
    
    # Display current panels
    st.write("Current Panels:")
    panel_df = pd.DataFrame(panels)
    st.dataframe(panel_df)
    
    # Add new panel
    st.write("Add New Panel:")
    col1, col2 = st.columns(2)
    
    with col1:
        new_title = st.text_input("Panel Title")
    
    with col2:
        new_url = st.text_input("Panel URL")
    
    new_height = st.slider("Panel Height", 200, 800, 400)
    
    if st.button("Add Panel"):
        if new_title and new_url:
            panels.append({
                'title': new_title,
                'url': new_url,
                'height': new_height
            })
            st.session_state.grafana_panels = panels
            st.success(f"Added panel: {new_title}")
            st.experimental_rerun()
        else:
            st.error("Title and URL are required")
    
    # Remove panel
    st.write("Remove Panel:")
    panel_titles = [p['title'] for p in panels]
    panel_to_remove = st.selectbox("Select Panel to Remove", panel_titles)
    
    if st.button("Remove Panel"):
        panels = [p for p in panels if p['title'] != panel_to_remove]
        st.session_state.grafana_panels = panels
        st.success(f"Removed panel: {panel_to_remove}")
        st.experimental_rerun()
    
    return panels