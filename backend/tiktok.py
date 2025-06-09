import json
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to determine sentiment label
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def tiktok_sentiment_analysis():
    # Load CSV
    df = pd.read_csv("data/ishowspeed_tiktok_hashtags.csv")

    # Clean and analyze
    df['text'] = df['text'].fillna('').astype(str)
    df['sentiment'] = df['text'].apply(get_sentiment)

    # Save to JSON
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    with open('output/tiktok_sentiment_results.json', 'w') as f:
        json.dump(sentiment_counts, f, indent=4)

    # Plot sentiment distribution
    # sentiment_counts = df['sentiment'].value_counts()
    # sentiment_counts.plot(kind='pie', title='Sentiment Distribution', color=['green', 'red', 'gray'])
    # plt.xlabel('Sentiment')
    # plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.show()

def main():
    tiktok_sentiment_analysis()

if __name__ == "__main__":
    main()
