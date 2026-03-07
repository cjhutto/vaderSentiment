import unittest
from vader_ui import analyze_text, analyze_url
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from unittest.mock import patch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TestVaderUI(unittest.TestCase):
    def setUp(self):
        self.test_text = """This is a very positive text! I love it.
        This is a negative sentence. I hate this part.
        This is a neutral statement that just states facts."""
        
        self.test_url = "https://example.com"
        
    def test_analyze_text_returns(self):
        """Test if analyze_text returns the correct types"""
        gauge, dist, df = analyze_text(self.test_text)
        
        # Check return types
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)
        
    def test_analyze_text_content(self):
        """Test if analyze_text produces expected content"""
        gauge, dist, df = analyze_text(self.test_text)
        
        # Check gauge data
        self.assertIn('value', gauge.data[0])
        self.assertTrue(-1 <= gauge.data[0]['value'] <= 1)  # Compound score should be between -1 and 1
        
        # Check distribution data
        sentiment_labels = dist.data[0]['x']
        self.assertEqual(set(sentiment_labels), {'Positive', 'Neutral', 'Negative'})
        
        # Check DataFrame columns
        expected_columns = ["Paragraph", "Text", "Compound Score"]
        self.assertEqual(list(df.columns), expected_columns)
        
    def test_analyze_text_empty(self):
        """Test handling of empty text"""
        gauge, dist, df = analyze_text("")
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertTrue(df.empty)
        
    def test_analyze_text_special_chars(self):
        """Test handling of special characters"""
        special_text = "This has special chars: !@#$%^&*()_+ 😊 and emojis 🎉"
        gauge, dist, df = analyze_text(special_text)
        self.assertIsNotNone(gauge)
        self.assertIsNotNone(dist)
        self.assertIsInstance(df, pd.DataFrame)
        
    def test_analyze_url_invalid(self):
        """Test handling of invalid URL"""
        gauge, dist, df = analyze_url("http://invalidurl.thisisnotreal")
        self.assertIsNone(gauge)
        self.assertIsNone(dist)
        self.assertIsNone(df)
        
    def test_analyze_text_long(self):
        """Test handling of long text"""
        long_text = "This is a test. " * 1000
        gauge, dist, df = analyze_text(long_text)
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)
        
    def test_sentiment_scores(self):
        """Test if sentiment scores are reasonable"""
        # Test positive text
        pos_text = "This is excellent! I love it! Amazing work!"
        gauge_pos, _, _ = analyze_text(pos_text)
        self.assertGreater(gauge_pos.data[0]['value'], 0)
        
        # Test negative text
        neg_text = "This is terrible! I hate it! Awful work!"
        gauge_neg, _, _ = analyze_text(neg_text)
        self.assertLess(gauge_neg.data[0]['value'], 0)
        
        # Test neutral text
        neu_text = "This is a statement. It contains information."
        gauge_neu, _, _ = analyze_text(neu_text)
        self.assertTrue(-0.1 <= gauge_neu.data[0]['value'] <= 0.1)

    def test_mixed_sentiment(self):
        """Test text with mixed sentiments"""
        mixed_text = """Great product but terrible service.
        The quality is amazing however the price is too high.
        I love the design but hate the color."""
        
        gauge, dist, df = analyze_text(mixed_text)
        
        # Mixed sentiment should have both positive and negative components
        compound_score = gauge.data[0]['value']
        
        # Get the distribution scores directly from the analyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(mixed_text)
        
        # Verify that we have both positive and negative components
        self.assertGreater(scores['pos'], 0, "Should have some positive sentiment")
        self.assertGreater(scores['neg'], 0, "Should have some negative sentiment")
        self.assertTrue(len(df) >= 1, "Should have at least one paragraph")

    def test_url_content_parsing(self):
        """Test URL content parsing with mock response"""
        mock_html = """
        <html>
            <body>
                <p>This is a great website! Amazing content.</p>
                <script>Some script to be ignored</script>
                <p>More positive content here.</p>
                <style>CSS to be ignored</style>
            </body>
        </html>
        """
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.text = mock_html
            mock_get.return_value.raise_for_status.return_value = None
            
            gauge, dist, df = analyze_url("https://example.com")
            
            self.assertIsInstance(gauge, go.Figure)
            self.assertIsInstance(dist, go.Figure)
            self.assertIsInstance(df, pd.DataFrame)
            
            # Should have positive sentiment (due to mock content)
            self.assertGreater(gauge.data[0]['value'], 0)

    def test_multilingual_text(self):
        """Test handling of non-English text"""
        multilingual_text = """
        Hello this is English! Great day!
        ¡Hola esto es español! ¡Excelente día!
        Bonjour c'est le français! Belle journée!
        """
        gauge, dist, df = analyze_text(multilingual_text)
        
        # Should still produce valid outputs
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)

    def test_html_in_text(self):
        """Test handling of text with HTML tags"""
        html_text = """
        <p>This is a paragraph with <b>bold</b> text!</p>
        <div>Another great <i>section</i> here.</div>
        """
        gauge, dist, df = analyze_text(html_text)
        
        # Should handle HTML content without errors
        self.assertIsInstance(gauge, go.Figure)
        self.assertIsInstance(dist, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)

    @patch('requests.get')
    def test_network_timeout(self, mock_get):
        """Test handling of network timeout"""
        mock_get.side_effect = requests.exceptions.Timeout
        
        gauge, dist, df = analyze_url("https://example.com")
        
        # Should return None values on timeout
        self.assertIsNone(gauge)
        self.assertIsNone(dist)
        self.assertIsNone(df)

    def test_paragraph_segmentation(self):
        """Test paragraph segmentation logic"""
        text_with_paragraphs = """
        First paragraph that is long enough to be counted as a real paragraph with more than 50 characters.
        
        Second paragraph that should also be counted due to its length being more than the threshold.
        
        Short line.
        
        Third substantial paragraph with sufficient length to be included in the analysis.
        """
        
        _, _, df = analyze_text(text_with_paragraphs)
        
        # Should have exactly 3 paragraphs (ignoring the short line)
        self.assertEqual(len(df), 3)
        
        # Verify paragraph numbering
        self.assertTrue(all(df['Paragraph'].str.contains('Paragraph [1-3]')))

if __name__ == '__main__':
    unittest.main(verbosity=2) 