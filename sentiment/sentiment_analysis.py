# sentiment/sentiment_analysis.py

from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0.2:
        return "Positive"
    elif analysis.sentiment.polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

