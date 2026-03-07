from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Test sentences
test_sentences = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience ever. Terrible service.",
    "The movie was okay, nothing special.",
    "The food was great, but the service was a bit slow.",
    "😊 This makes me so happy! 🌟",
]

# Analyze each sentence
for sentence in test_sentences:
    # Get the sentiment scores
    scores = analyzer.polarity_scores(sentence)
    
    print("\nSentence:", sentence)
    print("Sentiment scores:", scores)
    
    # Interpret the compound score
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    print("Overall sentiment:", sentiment) 