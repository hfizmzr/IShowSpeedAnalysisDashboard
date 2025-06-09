import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

def sentiment_over_time1(data):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()

    # Prepare dataframe
    all_comments = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        
        for comment in video.get("comments", []):
            score = analyzer.polarity_scores(comment)
            sentiment = (
                "positive" if score['compound'] >= 0.05
                else "negative" if score['compound'] <= -0.05
                else "neutral"
            )
            
            all_comments.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": pub_date,
                "comment": comment,
                "compound": score['compound'],
                "sentiment": sentiment
            })

    df = pd.DataFrame(all_comments)

    # Save to CSV (optional)
    df.to_csv("data/ishowspeed_comment_sentiment.csv", index=False)

    # Overview
    # print(df["sentiment"].value_counts())

    # Sentiment trend over time
    # Group and smooth with rolling average
    trend = df.groupby(["published_date", "sentiment"]).size().unstack().fillna(0)

    # Apply 3-day rolling average to smooth the lines
    smoothed_trend = trend.rolling(window=3, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=smoothed_trend)
    plt.title("Smoothed Sentiment Over Time")
    plt.xlabel("Published Date")
    plt.ylabel("Average Comment Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()

    # Save JSON version for JS use
    save_df_to_json(smoothed_trend, "smoothed_sentiment_trend.json")

def sentiment_over_time(data, skip_plotting=False):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()

    # Prepare dataframe
    all_comments = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        
        for comment in video.get("comments", []):
            score = analyzer.polarity_scores(comment)
            sentiment = (
                "positive" if score['compound'] >= 0.05
                else "negative" if score['compound'] <= -0.05
                else "neutral"
            )
            
            all_comments.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": pub_date,
                "comment": comment,
                "compound": score['compound'],
                "sentiment": sentiment
            })

    df = pd.DataFrame(all_comments)

    # Save to CSV (optional)
    df.to_csv("data/ishowspeed_comment_sentiment.csv", index=False)

    # Convert dates to monthly periods
    df['month'] = pd.to_datetime(df['published_date']).dt.to_period('M')
    
    # Group by month and sentiment, then calculate percentages
    monthly = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    
    # Calculate percentages for each sentiment
    monthly_percent = monthly.div(monthly.sum(axis=1), axis=0) * 100
    
    # Reorder columns to ensure proper stacking (positive on top)
    monthly_percent = monthly_percent[['positive', 'neutral', 'negative']]
    
    # Convert Period index back to datetime for plotting
    monthly_percent.index = monthly_percent.index.to_timestamp()

    # Plot stacked area chart (only if plotting is enabled)
    if not skip_plotting:
        plt.figure(figsize=(12, 6))
        plt.stackplot(
            monthly_percent.index,
            monthly_percent['negative'],
            monthly_percent['neutral'],
            monthly_percent['positive'],
            labels=['Negative', 'Neutral', 'Positive'],
            colors=['#F44336', '#FFC107', '#4CAF50']  # Green, Yellow, Red
        )
        
        plt.title("Monthly Sentiment Distribution (Stacked)")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()  # Commented out - causes server to hang in non-interactive environment
    
    # Save JSON version for JS use
    save_df_to_json(monthly_percent, "smoothed_sentiment_trend.json")

def sentiment_over_time_monthly(data, skip_plotting=False):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()

    all_comments = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        
        for comment in video.get("comments", []):
            score = analyzer.polarity_scores(comment)
            sentiment = (
                "positive" if score['compound'] >= 0.05
                else "negative" if score['compound'] <= -0.05
                else "neutral"
            )
            all_comments.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": pub_date,
                "comment": comment,
                "compound": score['compound'],
                "sentiment": sentiment
            })

    df = pd.DataFrame(all_comments)

    # Convert published_date to month period (e.g. 2024-05)
    df["month"] = pd.to_datetime(df["published_date"]).dt.to_period("M").astype(str)

    # Group by month and sentiment
    monthly_trend = df.groupby(["month", "sentiment"]).size().unstack().fillna(0)

    # Optional: sort index just in case
    monthly_trend = monthly_trend.sort_index()

    # Plot (only if plotting is enabled)
    if not skip_plotting:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_trend)
        plt.title("Monthly Sentiment Trend")
        plt.xlabel("Month")
        plt.ylabel("Comment Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

    # Save to CSV (optional)
    df.to_csv("data/ishowspeed_comment_sentiment.csv", index=False)

    # Save to JSON for frontend use
    save_df_to_json(monthly_trend, "monthly_sentiment_trend.json")



def vader_textblob(data, skip_plotting=False):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()

    # Store all comment data
    comments_data = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()

        for comment in video.get("comments", []):
            # VADER
            vader_score = analyzer.polarity_scores(comment)
            vader_sentiment = (
                "positive" if vader_score["compound"] >= 0.05
                else "negative" if vader_score["compound"] <= -0.05
                else "neutral"
            )

            # TextBlob
            blob_score = TextBlob(comment).sentiment.polarity
            blob_sentiment = (
                "positive" if blob_score > 0.05
                else "negative" if blob_score < -0.05
                else "neutral"
            )

            comments_data.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": pub_date,
                "comment": comment,
                "vader_compound": vader_score["compound"],
                "vader_sentiment": vader_sentiment,
                "textblob_polarity": blob_score,
                "textblob_sentiment": blob_sentiment
            })

    # Convert to DataFrame
    df = pd.DataFrame(comments_data)

    # Save to CSV
    df.to_csv("data/ishowspeed_sentiment_comparison.csv", index=False)

    # Sentiment comparison count
    # print("VADER Sentiment Counts:")
    # print(df["vader_sentiment"].value_counts(), "\n")

    # print("TextBlob Sentiment Counts:")
    # print(df["textblob_sentiment"].value_counts())

    # Plot comparison (only if plotting is enabled)
    if not skip_plotting:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        sns.countplot(data=df, x="vader_sentiment", hue="vader_sentiment", palette="Set2", legend=False)
        plt.title("VADER Sentiment")
        
        plt.subplot(1,2,2)
        sns.countplot(data=df, x="textblob_sentiment", hue="textblob_sentiment", palette="Set3", legend=False)
        plt.title("TextBlob Sentiment")

        plt.tight_layout()
        # plt.show()

    sentiment_summary = {
        "vader": df["vader_sentiment"].value_counts().to_dict(),
        "textblob": df["textblob_sentiment"].value_counts().to_dict()
    }

    # Save JSON
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "sentiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_summary, f, indent=4)
    print(f"Saved JSON to {summary_path}")


def save_df_to_json(df, filename):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    df.index = df.index.astype(str)  # Ensure date format is string
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="index"), f, indent=4)    
    print(f"Saved JSON to {os.path.join(output_dir, filename)}")

def top_positive_negative_comments(data, top_n=3):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()
    all_comments = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()

        for comment in video.get("comments", []):
            vader_score = analyzer.polarity_scores(comment)['compound']
            blob_score = TextBlob(comment).sentiment.polarity

            all_comments.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": str(pub_date),
                "comment": comment,
                "vader_compound": vader_score,
                "textblob_polarity": blob_score
            })

    df = pd.DataFrame(all_comments)

    # Get top positive/negative comments by VADER
    top_vader_pos = df.nlargest(top_n, 'vader_compound')[['video_title', 'comment', 'vader_compound']]
    top_vader_neg = df.nsmallest(top_n, 'vader_compound')[['video_title', 'comment', 'vader_compound']]

    # Get top positive/negative comments by TextBlob
    top_blob_pos = df.nlargest(top_n, 'textblob_polarity')[['video_title', 'comment', 'textblob_polarity']]
    top_blob_neg = df.nsmallest(top_n, 'textblob_polarity')[['video_title', 'comment', 'textblob_polarity']]

    result = {
        "top_vader_positive": top_vader_pos.to_dict(orient="records"),
        "top_vader_negative": top_vader_neg.to_dict(orient="records"),
        "top_textblob_positive": top_blob_pos.to_dict(orient="records"),
        "top_textblob_negative": top_blob_neg.to_dict(orient="records")
    }

    # Save to JSON
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "top_sentiment_comments.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Saved top sentiment comments to output/top_sentiment_comments.json")

def main():
    from data_collection import load_data
    data = load_data()
    sentiment_over_time(data)
    # vader_textblob(data)
    # sentiment_over_time_monthly(data)
    # top_positive_negative_comments(data)

if __name__ == "__main__":
    main()