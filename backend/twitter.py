import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from deep_translator import GoogleTranslator

def twitter_sentiment_analysis():
    df = pd.read_csv("data/keyword_ishowspeed_tweets.csv")

    df['language'] = df['text'].apply(lambda x: detect(x) if pd.notnull(x) else None)

    def translate_if_needed(text, lang):
        try:
            if lang != 'en':
                return GoogleTranslator(source='auto', target='en').translate(text)
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    df['translated_text'] = df.apply(lambda row: translate_if_needed(row['text'], row['language']), axis=1)

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        return analyzer.polarity_scores(text)["compound"]

    def label_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_score"] = df["translated_text"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["date"] = df["created_at"].dt.date

    # sns.countplot(data=df, x="sentiment_label", hue="sentiment_label", palette="coolwarm")
    # plt.title("Sentiment Distribution for IShowSpeed Mentions")
    # plt.xlabel("Sentiment")
    # plt.ylabel("Tweet Count")
    # plt.show()

    # trend = df.groupby(["date", "sentiment_label"]).size().unstack().fillna(0)
    # trend.plot(kind="line", marker='o', figsize=(12, 6))
    # plt.title("Sentiment Trend Over Time for IShowSpeed")
    # plt.xlabel("Date")
    # plt.ylabel("Tweet Count")
    # plt.legend(title="Sentiment")
    # plt.grid(True)
    # plt.show()

    df.to_csv("data/keyword_ishowspeed_tweets.csv", index=False)

    sentiment_counts = df['sentiment_label'].value_counts().to_dict()

    with open('output/twitter_sentiment_results.json', 'w') as f:
        json.dump(sentiment_counts, f, indent=4)

def twitter_content_analysis():
    df = pd.read_csv("data/ishowspeed_tweets.csv")
    df2 = pd.read_csv("data/mrbeast_tweets.csv")

    # Add a new empty column for content type
    df["content_type"] = ""
    df2["content_type"] = ""

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
    
    analyzer = SentimentIntensityAnalyzer()

    # Function to compute VADER sentiment score
    def get_sentiment_score(text):
        return analyzer.polarity_scores(text)["compound"]

    # Function to label sentiment
    def label_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Assuming df is your DataFrame and contains a 'text' column
    df["content_type"] = df["text"].apply(simplify_category)
    df["sentiment_score"] = df["text"].apply(get_sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    df2["content_type"] = df2["text"].apply(simplify_category)
    df2["sentiment_score"] = df2["text"].apply(get_sentiment_score)
    df2["sentiment_label"] = df2["sentiment_score"].apply(label_sentiment)

    df.to_csv("data/ishowspeed_tweets.csv", index=False)
    df2.to_csv("data/mrbeast_tweets.csv", index=False)

    output_json = df[['created_at', 'text', 'content_type', 'sentiment_label', 'sentiment_score']].to_dict(orient='records')
    with open('output/twitter_ishowspeed_content_results.json', 'w') as f:
        json.dump(output_json, f, indent=4)

    output_json = df2[['created_at', 'text', 'content_type', 'sentiment_label', 'sentiment_score']].to_dict(orient='records')
    with open('output/twitter_mrbeast_content_results.json', 'w') as f:
        json.dump(output_json, f, indent=4)

def twitter_comparison():
    # Load both datasets
    df_speed = pd.read_csv("data/ishowspeed_tweets.csv")
    df_beast = pd.read_csv("data/mrbeast_tweets.csv")

    columns_needed = [
        "content_type",
        "views_count",
        "like_count",
        "retweet_count",
        "reply_count",
        "quote_count",
    ]

    df_speed_filtered = df_speed[columns_needed].copy()
    df_beast_filtered = df_beast[columns_needed].copy()

    # Add creator column after filtering
    df_speed_filtered["creator"] = "IShowSpeed"
    df_beast_filtered["creator"] = "MrBeast"

    combined = pd.concat([df_speed_filtered, df_beast_filtered], ignore_index=True)

    # Split content_type by comma and explode into multiple rows
    combined["content_type"] = combined["content_type"].str.split(r",\s*")  # split and remove extra spaces
    combined = combined.explode("content_type")

    # Strip whitespace from labels
    combined["content_type"] = combined["content_type"].str.strip()

    # Group and calculate mean engagement
    metrics = ["views_count", "like_count", "reply_count", "retweet_count", "quote_count"]
    avg_engagement = combined.groupby(["creator", "content_type"])[metrics].mean().reset_index()

    # Pretty-print JSON (not line-delimited)
    avg_engagement.to_json("output/twitter_avg_engagement.json", orient="records", indent=4)

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=avg_engagement, x="content_type", y="like_count", hue="creator", palette="Accent")
    # plt.title("Average Likes by Content Type")
    # plt.xticks(rotation=45)
    # plt.ylabel("Average Likes")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=avg_engagement, x="content_type", y="views_count", hue="creator", palette="Accent")
    # plt.title("Average Views by Content Type")
    # plt.xticks(rotation=45)
    # plt.ylabel("Average Likes")
    # plt.tight_layout()
    # plt.show()

def main():
    twitter_sentiment_analysis()
    twitter_content_analysis()
    twitter_comparison()

if __name__ == "__main__":
    main()
