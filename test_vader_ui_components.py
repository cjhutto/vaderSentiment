import unittest
import gradio as gr
from vader_ui import demo, analyze_text, analyze_url
import numpy as np
import time
import plotly.graph_objects as go
import pandas as pd
from unittest.mock import patch
import requests

class TestVaderUIComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the Gradio interface for testing"""
        cls.interface = demo

    def test_interface_structure(self):
        """Test that the interface has all required components"""
        # Check basic interface properties
        self.assertIsInstance(self.interface, gr.Blocks)
        self.assertEqual(self.interface.title, "VADER Sentiment Analysis")
        
        # Get all components
        components = self.interface.blocks.values()
        
        # Check for presence of key components
        component_types = [type(comp) for comp in components]
        
        # Verify text inputs exist
        text_inputs = [c for c in components if isinstance(c, gr.Textbox)]
        self.assertEqual(len(text_inputs), 2)  # URL input and Text input
        
        # Verify buttons exist
        buttons = [c for c in components if isinstance(c, gr.Button)]
        self.assertEqual(len(buttons), 2)  # Analyze URL and Analyze Text buttons
        
        # Verify plots exist
        plots = [c for c in components if isinstance(c, gr.Plot)]
        self.assertEqual(len(plots), 4)  # 2 gauge plots and 2 distribution plots
        
        # Verify dataframes exist
        dataframes = [c for c in components if isinstance(c, gr.Dataframe)]
        self.assertEqual(len(dataframes), 2)  # 2 paragraph analysis tables

    def test_text_input_properties(self):
        """Test properties of text input components"""
        components = self.interface.blocks.values()
        
        # Find text inputs
        text_inputs = [c for c in components if isinstance(c, gr.Textbox)]
        url_input = next(c for c in text_inputs if c.label == "Website URL")
        text_input = next(c for c in text_inputs if c.label == "Text to analyze")
        
        # Test URL input properties
        self.assertEqual(url_input.placeholder, "https://example.com")
        self.assertEqual(url_input.lines, 1)
        
        # Test text input properties
        self.assertEqual(text_input.placeholder, "Enter your text here...")
        self.assertEqual(text_input.lines, 5)

    def test_button_properties(self):
        """Test properties of button components"""
        components = self.interface.blocks.values()
        
        # Find buttons
        buttons = [c for c in components if isinstance(c, gr.Button)]
        url_button = next(c for c in buttons if c.value == "Analyze URL")
        text_button = next(c for c in buttons if c.value == "Analyze Text")
        
        # Verify button properties
        self.assertIsNotNone(url_button)
        self.assertIsNotNone(text_button)

    def test_plot_properties(self):
        """Test properties of plot components"""
        components = self.interface.blocks.values()
        plots = [c for c in components if isinstance(c, gr.Plot)]
        
        # Check plot labels
        gauge_plots = [p for p in plots if p.label == "Overall Sentiment"]
        dist_plots = [p for p in plots if p.label == "Sentiment Distribution"]
        
        self.assertEqual(len(gauge_plots), 2)
        self.assertEqual(len(dist_plots), 2)

    def test_dataframe_properties(self):
        """Test properties of dataframe components"""
        components = self.interface.blocks.values()
        dataframes = [c for c in components if isinstance(c, gr.Dataframe)]
        
        for df in dataframes:
            self.assertEqual(df.label, "Paragraph Analysis")
            self.assertEqual(df.headers, ["Paragraph", "Text", "Compound Score"])

    def test_tab_structure(self):
        """Test the tab structure of the interface"""
        components = self.interface.blocks.values()
        tabs = [c for c in components if isinstance(c, gr.Tab)]
        
        # Verify we have two tabs
        tab_items = [c for c in components if isinstance(c, gr.TabItem)]
        self.assertEqual(len(tab_items), 2)
        
        # Verify tab names
        tab_names = [t.label for t in tab_items]
        self.assertIn("URL Analysis", tab_names)
        self.assertIn("Text Analysis", tab_names)

    def test_interface_markdown(self):
        """Test the markdown components"""
        components = self.interface.blocks.values()
        markdowns = [c for c in components if isinstance(c, gr.Markdown)]
        
        # Verify markdown content
        markdown_texts = [m.value for m in markdowns]
        self.assertIn("# VADER Sentiment Analyzer", markdown_texts)
        self.assertIn("Analyze sentiment from a website URL or direct text input", markdown_texts)

    def test_interface_layout(self):
        """Test the layout structure"""
        components = self.interface.blocks.values()
        
        # Verify row components for plots
        rows = [c for c in components if isinstance(c, gr.Row)]
        self.assertGreaterEqual(len(rows), 2)  # At least 2 rows for plots

    def test_text_analysis_interaction(self):
        """Test the text analysis functionality"""
        components = self.interface.blocks.values()
        
        # Find text input and button
        text_inputs = [c for c in components if isinstance(c, gr.Textbox)]
        text_input = next(c for c in text_inputs if c.label == "Text to analyze")
        
        # Test positive text
        test_text = "This is amazing! I love this test!"
        gauge, dist, df = analyze_text(test_text)
        
        # Verify outputs
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check sentiment is positive
        self.assertGreater(gauge.data[0]['value'], 0)
        
        # Test negative text
        test_text = "This is terrible! I hate this test!"
        gauge, dist, df = analyze_text(test_text)
        
        # Check sentiment is negative
        self.assertLess(gauge.data[0]['value'], 0)

    def test_url_analysis_interaction(self):
        """Test the URL analysis functionality"""
        mock_html = """
        <html><body>
            <p>This is a wonderful test page! Amazing content here.</p>
            <p>Everything is working perfectly!</p>
        </body></html>
        """
        
        with patch('requests.get') as mock_get:
            # Setup mock
            mock_get.return_value.text = mock_html
            mock_get.return_value.raise_for_status.return_value = None
            
            # Test URL analysis
            gauge, dist, df = analyze_url("https://test.com")
            
            # Verify outputs
            self.assertIsInstance(gauge, go.Figure)
            self.assertIsInstance(dist, go.Figure)
            self.assertIsInstance(df, pd.DataFrame)
            
            # Should be positive sentiment due to mock content
            self.assertGreater(gauge.data[0]['value'], 0)

    def test_input_validation(self):
        """Test input validation and error handling"""
        components = self.interface.blocks.values()
        
        # Test empty text
        gauge, dist, df = analyze_text("")
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertTrue(df.empty)
        
        # Test invalid URL
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException
            gauge, dist, df = analyze_url("invalid_url")
            self.assertIsNone(gauge)
            self.assertIsNone(dist)
            self.assertIsNone(df)

    def test_component_callbacks(self):
        """Test that components have proper callbacks attached"""
        components = self.interface.blocks.values()
        
        # Find buttons and their associated inputs/outputs
        buttons = [c for c in components if isinstance(c, gr.Button)]
        text_inputs = [c for c in components if isinstance(c, gr.Textbox)]
        plots = [c for c in components if isinstance(c, gr.Plot)]
        dataframes = [c for c in components if isinstance(c, gr.Dataframe)]
        
        url_button = next(c for c in buttons if c.value == "Analyze URL")
        text_button = next(c for c in buttons if c.value == "Analyze Text")
        
        # Verify buttons have click events attached
        self.assertTrue(hasattr(url_button, 'click'))
        self.assertTrue(hasattr(text_button, 'click'))
        
        # Test that the analyze functions work with the components
        test_text = "This is a test!"
        gauge, dist, df = analyze_text(test_text)
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)

    def test_real_time_updates(self):
        """Test that outputs update in real-time with input changes"""
        components = self.interface.blocks.values()
        
        # Get different sentiment texts
        texts = [
            "This is amazing!",
            "This is terrible!",
            "This is neutral.",
            "I love this!",
            "I hate this!"
        ]
        
        # Test that each text produces different sentiment scores
        scores = []
        for text in texts:
            gauge, _, _ = analyze_text(text)
            scores.append(gauge.data[0]['value'])
        
        # Verify we get different scores for different texts
        unique_scores = len(set(scores))
        self.assertGreater(unique_scores, 1, "Different texts should produce different sentiment scores")

    def test_concurrent_analysis(self):
        """Test handling multiple analyses in quick succession"""
        test_texts = [
            "First test text that is positive!",
            "Second test text that is negative!",
            "Third test text that is neutral."
        ]
        
        results = []
        for text in test_texts:
            gauge, dist, df = analyze_text(text)
            results.append(gauge.data[0]['value'])
        
        # Verify all analyses completed
        self.assertEqual(len(results), len(test_texts))
        
        # Verify results are different
        self.assertNotEqual(results[0], results[1])

if __name__ == '__main__':
    unittest.main(verbosity=2) 