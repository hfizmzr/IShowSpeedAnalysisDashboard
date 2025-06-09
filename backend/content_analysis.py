import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

def content(data):
    videos = data['latest_videos']
    df = pd.DataFrame(videos)

    # Define simplified categories
    simplified_categories = {
    'Gaming': [
        'play', 'playing', 'game', 'fifa', 'fortnite', 'minecraft', 'gta', 'call of duty', 'cod', 'valorant',
        'apex', 'roblox', 'match', 'gameplay', 'ranked', 'level', 'boss fight', 'kill', 'noob', 'pro', 'stream', 'live'
    ],
    'Memes': [
        'meme', 'funny', 'laugh', 'haha', 'comedy', 'skit', 'joke', 'humor', 'parody', 'relatable', 'dark humor',
        'troll', 'sarcasm', 'goofy', 'lol', 'meme compilation', 'meme review'
    ],
    'Challenges / IRL': [
        'irl', 'real life', 'vlog', '24h', '24 hours', 'daily life', 'adventure', 'prank', 'challenge', 'in',
        'public', 'school', 'mall', 'store', 'eating', 'shopping', 'trying', 'experiment', 'gone wrong', 'overnight',
        'vs', 'challenge accepted', 'social experiment', 'with strangers', 'in'
    ],
    'Music': [
        'music', 'rap', 'song', 'freestyle', 'singing', 'track', 'lyrics', 'beat', 'cover', 'official video',
        'instrumental', 'remix', 'album', 'chorus', 'verse', 'studio', 'mixtape', 'hook', 'banger'
    ],
    'Reactions': [
        'react', 'reaction', 'watching', 'first time', 'response', 'opinion', 'review', 'seeing', 'listening to',
        'responding to', 'try not to', 'watch this', 'cringe', 'shocking moment', 'blind reaction'
    ],
    'Viral moments / Clickbait': [
        'crazy', 'wtf', 'omg', 'insane', 'viral', 'shocking', 'unbelievable', 'emotional', 'wild',
        'believe', 'caught on camera', 'exposed', 'gone viral', 'drama', 'intense', 'insanity',
        'breaking', 'clickbait', 'scandal', 'leak', 'revealed', 'must see', 'wtf just happened'
    ]
    }


    def simplify_category(title):
        title = title.lower()
        for category, keywords in simplified_categories.items():
            if any(keyword in title for keyword in keywords):
                return category
        return 'Other'

    df['content_type'] = df['title'].apply(simplify_category)

    # Load sentiment CSV
    sentiment_df = pd.read_csv('data/ishowspeed_sentiment_comparison.csv')

    # Ensure video_id is string for both
    df['video_id'] = df['video_id'].astype(str)
    sentiment_df['video_id'] = sentiment_df['video_id'].astype(str)

    # Count sentiment per video
    sentiment_counts = sentiment_df.groupby(['video_id', 'vader_sentiment']).size().unstack(fill_value=0).reset_index()

    # Merge sentiment back to main DataFrame
    df = df.merge(sentiment_counts, on='video_id', how='left')
    df[['positive', 'neutral', 'negative']] = df[['positive', 'neutral', 'negative']].fillna(0)

    # --------------------------
    # Engagement Summary
    # --------------------------
    agg = df.groupby('content_type').agg(
        number_of_videos=('video_id', 'count'),
        avg_views=('view_count', 'mean'),
        avg_likes=('like_count', 'mean'),
        avg_comments=('comment_count', 'mean')
    ).reset_index()

    print("\nEngagement by Content Type:\n", agg)

    # --------------------------
    # Visualization
    # --------------------------

    # 1. Bar Chart - Engagement
    # plt.figure(figsize=(10,6))
    # agg_plot = agg.sort_values("avg_views", ascending=False)
    # sns.barplot(x='avg_views', y='content_type', data=agg_plot, palette="viridis")
    # plt.title("Average Views by Content Type")
    # plt.xlabel("Average Views")
    # plt.ylabel("Content Type")
    # plt.tight_layout()
    # plt.show()

    # 2. Sentiment - Stacked Bar Chart
    sentiment_by_type = df.groupby('content_type')[['positive', 'neutral', 'negative']].sum()
    sentiment_by_type = sentiment_by_type.loc[sentiment_by_type.sum(axis=1) > 0]  # drop types with no sentiment

    # sentiment_by_type.plot(kind='barh', stacked=True, figsize=(10,6), colormap='coolwarm')
    # plt.title("Total Sentiment Counts by Content Type")
    # plt.xlabel("Number of Comments")
    # plt.ylabel("Content Type")
    # plt.tight_layout()
    # plt.show()

    # Ensure output dir
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save engagement summary (avg_views, avg_likes, etc.)
    agg_json = agg.to_dict(orient="records")
    with open(os.path.join(output_dir, "content_engagement.json"), "w", encoding="utf-8") as f:
        json.dump(agg_json, f, indent=4)

    # Save sentiment per content type (positive, neutral, negative)
    sentiment_json = sentiment_by_type.reset_index().to_dict(orient="records")
    with open(os.path.join(output_dir, "content_sentiment.json"), "w", encoding="utf-8") as f:
        json.dump(sentiment_json, f, indent=4)

    print("Saved content_engagement.json and content_sentiment.json in /output")

    generate_extra_content_insights(agg, sentiment_by_type, output_dir)

def generate_extra_content_insights(agg, sentiment_by_type, output_dir):
    # --------------------------
    # Extra Metrics Calculation
    # --------------------------
    agg['like_to_view_ratio'] = agg['avg_likes'] / agg['avg_views']
    agg['comment_to_view_ratio'] = agg['avg_comments'] / agg['avg_views']

    sentiment_pct = sentiment_by_type.copy()
    sentiment_totals = sentiment_pct.sum(axis=1)
    sentiment_pct['positive_pct'] = sentiment_pct['positive'] / sentiment_totals
    sentiment_pct['neutral_pct'] = sentiment_pct['neutral'] / sentiment_totals
    sentiment_pct['negative_pct'] = sentiment_pct['negative'] / sentiment_totals

    # Save sentiment percentages
    sentiment_pct_cleaned = sentiment_pct[['positive_pct', 'neutral_pct', 'negative_pct']].reset_index()
    sentiment_pct_cleaned.to_json(os.path.join(output_dir, "content_sentiment_percentage.json"), orient="records", indent=4)

    # --------------------------
    # Visualizations (Extras)
    # --------------------------

    # 1. Like-to-View Ratio
    # plt.figure(figsize=(10,6))
    # sns.barplot(x='like_to_view_ratio', y='content_type', data=agg.sort_values("like_to_view_ratio", ascending=False), palette="crest")
    # plt.title("Like-to-View Ratio by Content Type")
    # plt.xlabel("Like-to-View Ratio")
    # plt.ylabel("Content Type")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "like_to_view_ratio.png"))
    # plt.close()

    # 2. Sentiment % Breakdown (Horizontal Stacked Bar)
    # plt.figure(figsize=(10,6))
    # sentiment_pct_cleaned.set_index('content_type')[['positive_pct', 'neutral_pct', 'negative_pct']].plot(
    #     kind='barh', stacked=True, colormap='RdYlGn', figsize=(10,6)
    # )
    # plt.title("Sentiment Percentages by Content Type")
    # plt.xlabel("Proportion of Sentiment")
    # plt.ylabel("Content Type")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "sentiment_percentage_stacked.png"))
    # plt.close()

    # 3. Likes vs Comments Scatter
    # plt.figure(figsize=(8,6))
    # sns.scatterplot(data=agg, x="avg_likes", y="avg_comments", hue="content_type", s=100)
    # plt.title("Likes vs Comments per Content Type")
    # plt.xlabel("Average Likes")
    # plt.ylabel("Average Comments")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "likes_vs_comments.png"))
    # plt.close()

    print("Additional insights and visualizations saved in /output")


def main():
    from data_collection import load_data
    data = load_data()
    content(data)

if __name__ == "__main__":
    main()