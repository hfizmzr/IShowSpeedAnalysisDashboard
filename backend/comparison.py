import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# ---------- Channel Summary ----------
def get_channel_summary(data, filename):
    channel = data["channel_data"]
    videos = data["latest_videos"]

    top_videos = videos[:5] if len(videos) >= 5 else videos
    avg_likes = sum(v["like_count"] for v in top_videos) / len(top_videos) if top_videos else 0
    avg_comments = sum(v["comment_count"] for v in top_videos) / len(top_videos) if top_videos else 0

    avg_engagement_per_video = avg_likes + avg_comments
    subscribers = channel["subscriber_count"] if channel["subscriber_count"] else 1  # Avoid divide by zero

    engagement_rate = (avg_engagement_per_video / subscribers) * 100

    summary = {
        "name": channel["channel_name"],
        "subscribers": subscribers,
        "total_views": channel["total_view_count"],
        "video_count": channel["video_count"],
        "average_views_per_video": channel["total_view_count"] / channel["video_count"] if channel["video_count"] else 0,
        "engagement_rate (%)": round(engagement_rate, 3)
    }

    save_to_json(summary, filename)
    return summary

# ---------- Engagement Stats ----------
def get_engagement_stats(data, filename="engagement_stats.json", top_n=5):
    videos = data["latest_videos"][:top_n]
    stats = {
        "average_views": sum(v["view_count"] for v in videos) / top_n,
        "average_likes": sum(v["like_count"] for v in videos) / top_n,
        "average_comments": sum(v["comment_count"] for v in videos) / top_n
    }
    save_to_json(stats, filename)
    return stats

# ---------- Posting Frequency ----------
def get_posting_frequency(data, filename="posting_freq.json", days=30):
    cutoff = datetime.now() - timedelta(days=days)
    count = sum(datetime.strptime(v["published_at"], "%Y-%m-%dT%H:%M:%SZ") > cutoff for v in data["latest_videos"])
    save_to_json({"videos_last_30_days": count}, filename)
    return count

# ---------- Sentiment Analysis ----------
def vader(data, filename):
    videos = data["latest_videos"]
    analyzer = SentimentIntensityAnalyzer()

    comments_data = []

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        pub_date = datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ").date()

        for comment in video.get("comments", []):
            vader_score = analyzer.polarity_scores(comment)
            vader_sentiment = (
                "positive" if vader_score["compound"] >= 0.05
                else "negative" if vader_score["compound"] <= -0.05
                else "neutral"
            )

            comments_data.append({
                "video_id": video_id,
                "video_title": title,
                "published_date": str(pub_date),
                "comment": comment,
                "vader_compound": vader_score["compound"],
                "vader_sentiment": vader_sentiment,
            })

    df = pd.DataFrame(comments_data)

    # Plot and save the figure
    # plt.figure(figsize=(12, 5))
    # sns.countplot(data=df, x="vader_sentiment", hue="vader_sentiment", palette="Set2", legend=False)
    # plt.title("VADER Sentiment")
    # plt.tight_layout()
    # output_dir = os.path.join(os.getcwd(), "output")
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, f"{filename.replace('.json', '_plot.png')}"))
    # plt.close()

    sentiment_summary = {
        "vader": df["vader_sentiment"].value_counts().to_dict(),
    }

    save_to_json(sentiment_summary, filename)

# ---------- Save to JSON ----------
def save_to_json(data, filename):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON to {path}")

# ---------- Main ----------
def main():
    from data_collection import load_data_from_file
    from analyze import load_data_from_file_df, save_series_to_json, analyze_views

    ishowspeed_data = load_data_from_file('data/ishowspeed_latest_500.json')
    mrbeast_data = load_data_from_file('data/mrbeast_latest_500.json')
    kaicenat_data = load_data_from_file('data/kaicenat_latest_500.json')

    # Channel Summary
    get_channel_summary(ishowspeed_data, 'ishowspeed_summary.json')
    get_channel_summary(mrbeast_data, 'mrbeast_summary.json')
    get_channel_summary(kaicenat_data, 'kaicenat_summary.json')

    # Engagement Stats
    get_engagement_stats(ishowspeed_data, 'ishowspeed_engagement.json')
    get_engagement_stats(mrbeast_data, 'mrbeast_engagement.json')
    get_engagement_stats(kaicenat_data, 'kaicenat_engagement.json')

    # Posting Frequency
    get_posting_frequency(ishowspeed_data, 'ishowspeed_posting_freq.json')
    get_posting_frequency(mrbeast_data, 'mrbeast_posting_freq.json')
    get_posting_frequency(kaicenat_data, 'kaicenat_posting_freq.json')

    # Sentiment Analysis
    vader(ishowspeed_data, 'ishowspeed_vader.json')
    vader(mrbeast_data, 'mrbeast_vader.json')
    vader(kaicenat_data, 'kaicenat_vader.json')

    # Monthly Views
    df, channel_data = load_data_from_file_df('data/mrbeast_latest_500.json')
    monthly_views, quarterly_views, monthly_growth, quarterly_growth = analyze_views(df)
    save_series_to_json(monthly_views, "mrbeast_monthly_views.json")

    df, channel_data = load_data_from_file_df('data/kaicenat_latest_500.json')
    monthly_views, quarterly_views, monthly_growth, quarterly_growth = analyze_views(df)
    save_series_to_json(monthly_views, "kaicenat_monthly_views.json")

if __name__ == "__main__":
    main()
