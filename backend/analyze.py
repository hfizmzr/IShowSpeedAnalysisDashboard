import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime
import os
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

def load_data(file_path="data/ishowspeed_latest_500.json"):
    # Load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    videos = data['latest_videos']
    channel_data = data['channel_data']
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(videos)
    
    # Convert publish date to datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    # Sort by publish date
    df = df.sort_values('published_at')
    
    # Add month and quarter columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['month'] = df['published_at'].dt.to_period('M')
        df['quarter'] = df['published_at'].dt.to_period('Q')
    
    return df, channel_data

def load_data_from_file_df(file_path):
    # Load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    videos = data['latest_videos']
    channel_data = data['channel_data']
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(videos)
    
    # Convert publish date to datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    # Sort by publish date
    df = df.sort_values('published_at')
    
    # Add month and quarter columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['month'] = df['published_at'].dt.to_period('M')
        df['quarter'] = df['published_at'].dt.to_period('Q')
    
    return df, channel_data

def analyze_views(df):
    # Aggregate total views by month and quarter
    monthly_views = df.groupby('month')['view_count'].sum()
    quarterly_views = df.groupby('quarter')['view_count'].sum()
    
    print("Monthly views:\n", monthly_views)
    print("Quarterly views:\n", quarterly_views)
    
    # Calculate monthly % growth
    monthly_growth = monthly_views.pct_change().fillna(0) * 100
    print("Monthly growth rates (%):\n", monthly_growth)
    
    # Calculate quarterly % growth
    quarterly_growth = quarterly_views.pct_change().fillna(0) * 100
    print("Quarterly growth rates (%):\n", quarterly_growth)
    
    # Define spike threshold, e.g. growth > 50%
    spikes = monthly_growth[monthly_growth > 50]
    print("Major spikes in monthly growth (>50%):\n", spikes)
    
    return monthly_views, quarterly_views, monthly_growth, quarterly_growth

def analyze_likes(df):
    # Aggregate total likes by month and quarter
    monthly_likes = df.groupby('month')['like_count'].sum()
    quarterly_likes = df.groupby('quarter')['like_count'].sum()
    
    print("Monthly likes:\n", monthly_likes)
    print("Quarterly likes:\n", quarterly_likes)
    
    # Calculate monthly % growth for likes
    monthly_growth_likes = monthly_likes.pct_change().fillna(0) * 100
    print("Monthly growth rates for likes (%):\n", monthly_growth_likes)
    
    return monthly_likes, quarterly_likes, monthly_growth_likes

def plot_smoothed_trend(data_series, title, ylabel, color='skyblue', label=None):
    # Monthly smoothing using spline interpolation
    x = np.arange(len(data_series))
    y = data_series.values
    x_new = np.linspace(x.min(), x.max(), 300)  # More points for smoothness
    spline = make_interp_spline(x, y)
    y_smooth = spline(x_new)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_new, y_smooth, color=color, label=label if label else 'Trend')
    plt.title(title)
    plt.xlabel('Time Period')
    plt.ylabel(ylabel)
    plt.xticks(ticks=x, labels=data_series.index.astype(str), rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.show()

# Save series to JSON
def save_series_to_json(series, filename, label_name="label", value_name="value"):
    # Convert Series to list of dicts for JS
    data = [{label_name: str(idx), value_name: int(val)} for idx, val in series.items()]
    
    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved JSON to {os.path.join(output_dir, filename)}")

def analyze_engagement_rate(df):
    # Avoid division by zero
    df['engagement_rate'] = ((df['like_count'] + df['comment_count']) / df['view_count'].replace(0, np.nan)) * 100
    df['engagement_rate'] = df['engagement_rate'].fillna(0)
    
    # Monthly average engagement rate
    monthly_er = df.groupby('month')['engagement_rate'].mean()
    print("Monthly average engagement rates (%):\n", monthly_er)

    # Total engagement rate across all videos
    total_likes = df['like_count'].sum()
    total_comments = df['comment_count'].sum()
    total_views = df['view_count'].sum()
    
    total_engagement_rate = ((total_likes + total_comments) / total_views) * 100 if total_views > 0 else 0
    print(f"Total engagement rate across all videos: {total_engagement_rate:.2f}%")

    return monthly_er, total_engagement_rate

def get_top_videos(df, n=3):
    top_videos = df.sort_values(by='view_count', ascending=False).head(n)
    
    result = []
    for _, row in top_videos.iterrows():
        video_id = row['video_id']
        result.append({
            "title": row['title'],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "view_count": int(row['view_count']),
            "like_count": int(row['like_count']),
            "comment_count": int(row['comment_count']),
            "published_at": row['published_at'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "top_3_videos.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    print("Saved top 3 videos to output/top_3_videos.json")
    
    return result


def main():
    # Load and prepare data
    df, channel_data = load_data()
    
    # Analyze views
    monthly_views, quarterly_views, monthly_growth, quarterly_growth = analyze_views(df)
    
    # Analyze likes
    monthly_likes, quarterly_likes, monthly_growth_likes = analyze_likes(df)

    # Save to JSON
    save_series_to_json(monthly_views, "monthly_views.json")
    save_series_to_json(monthly_growth, "monthly_growth.json", value_name="growth")
    save_series_to_json(monthly_likes, "monthly_likes.json")
    save_series_to_json(monthly_growth_likes, "monthly_growth_likes.json", value_name="growth")

    # Analyze engagement rate
    monthly_er, total_er = analyze_engagement_rate(df)
    # Save engagement rate to JSON
    save_series_to_json(monthly_er, "monthly_engagement_rate.json", value_name="engagement_rate")
    
    with open("output/total_engagement_rate.json", "w", encoding="utf-8") as f:
        json.dump({"total_engagement_rate": round(total_er, 2)}, f, indent=4)


    # Get top 3 videos by views
    get_top_videos(df)
    
    # Plot smoothed trends
    plot_smoothed_trend(monthly_views, 'Smoothed Monthly Views (Spline)', 'View Count', 'skyblue', 'Monthly Views')
    plot_smoothed_trend(monthly_growth, 'Smoothed Monthly Growth (Spline)', 'Growth Rate (%)', 'skyblue', 'Monthly Growth')
    plot_smoothed_trend(monthly_likes, 'Smoothed Monthly Likes (Spline)', 'Like Count', 'skyblue', 'Monthly Likes')
    plot_smoothed_trend(monthly_growth_likes, 'Smoothed Monthly Likes Growth (Spline)', 'Growth Rate (%)', 'skyblue', 'Monthly Likes Growth')

if __name__ == "__main__":
    main()