import json
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def ig_sentiment_analysis():
    df = pd.read_csv("data/ig_top_comments.csv")

    analyzer = SentimentIntensityAnalyzer()

    # Clean data
    df = df.dropna(subset=["text"])  # Remove rows where text is NaN
    df["text"] = df["text"].astype(str)  # Ensure all text entries are strings

    def get_sentiment(text):
        return analyzer.polarity_scores(text)["compound"]

    def label_sentiment(score):
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_score"] = df["text"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    df.to_csv("data/ig_top_comments.csv", index=False)
    
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()

    with open('output/ig_sentiment_results.json', 'w') as f:
        json.dump(sentiment_counts, f, indent=4)

def main():
    ig_sentiment_analysis()

if __name__ == "__main__":
    main()