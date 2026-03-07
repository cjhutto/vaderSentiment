import gradio as gr
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def analyze_text(text):
    try:
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()
        
        # Get overall sentiment scores
        scores = analyzer.polarity_scores(text)
        
        # Create sentiment gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = scores['compound'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.05], 'color': "red"},
                    {'range': [-0.05, 0.05], 'color': "gray"},
                    {'range': [0.05, 1], 'color': "green"}
                ],
            }
        ))
        
        # Create sentiment distribution bar chart
        sentiment_dist = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [scores['pos'], scores['neu'], scores['neg']]
        })
        
        fig_dist = px.bar(sentiment_dist, x='Sentiment', y='Score',
                         color='Sentiment',
                         color_discrete_map={
                             'Positive': 'green',
                             'Neutral': 'gray',
                             'Negative': 'red'
                         },
                         title='Sentiment Distribution')
        fig_dist.update_traces(x=['Positive', 'Neutral', 'Negative'])
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
        para_sentiments = []
        
        for i, para in enumerate(paragraphs[:5], 1):
            para_scores = analyzer.polarity_scores(para)
            para_sentiments.append({
                'Paragraph': f'Paragraph {i}',
                'Text': para[:100] + "...",
                'Compound Score': para_scores['compound']
            })
        
        # Create paragraph analysis table
        df_paragraphs = pd.DataFrame(para_sentiments) if para_sentiments else pd.DataFrame(columns=["Paragraph", "Text", "Compound Score"])
        
        return fig_gauge, fig_dist, df_paragraphs
    except Exception as e:
        return None, None, None

def analyze_url(url):
    try:
        # Fetch and parse website content
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return analyze_text(text)
    except Exception as e:
        return None, None, None

# Create Gradio interface
with gr.Blocks(title="VADER Sentiment Analysis") as demo:
    gr.Markdown("# VADER Sentiment Analyzer")
    gr.Markdown("Analyze sentiment from a website URL or direct text input")
    
    with gr.Tabs():
        with gr.TabItem("URL Analysis"):
            url_input = gr.Textbox(label="Website URL", placeholder="https://example.com")
            analyze_url_button = gr.Button("Analyze URL")
            
            with gr.Row():
                url_gauge_plot = gr.Plot(label="Overall Sentiment")
                url_dist_plot = gr.Plot(label="Sentiment Distribution")
            
            url_paragraph_table = gr.Dataframe(
                headers=["Paragraph", "Text", "Compound Score"],
                label="Paragraph Analysis"
            )
            
            analyze_url_button.click(
                analyze_url,
                inputs=[url_input],
                outputs=[url_gauge_plot, url_dist_plot, url_paragraph_table]
            )
        
        with gr.TabItem("Text Analysis"):
            text_input = gr.Textbox(label="Text to analyze", placeholder="Enter your text here...", lines=5)
            analyze_text_button = gr.Button("Analyze Text")
            
            with gr.Row():
                text_gauge_plot = gr.Plot(label="Overall Sentiment")
                text_dist_plot = gr.Plot(label="Sentiment Distribution")
            
            text_paragraph_table = gr.Dataframe(
                headers=["Paragraph", "Text", "Compound Score"],
                label="Paragraph Analysis"
            )
            
            analyze_text_button.click(
                analyze_text,
                inputs=[text_input],
                outputs=[text_gauge_plot, text_dist_plot, text_paragraph_table]
            )

if __name__ == "__main__":
    demo.launch() 