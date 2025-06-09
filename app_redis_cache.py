import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from googleapiclient.discovery import build
import time
from functools import wraps
import pickle
import hashlib
import redis
from typing import Optional, Any

# Import all the modules
from backend.data_collection import collect_youtube_data, get_channel_info, collect_video_comments, load_data_from_file, collect_for_creator
from backend.analyze import load_data, analyze_views, analyze_likes, plot_smoothed_trend, save_series_to_json, load_data_from_file_df
from backend.content_analysis import content
from backend.sentiment_analysis import sentiment_over_time, vader_textblob, sentiment_over_time_monthly,top_positive_negative_comments
from backend.prediction import forecast_views, forecast_next_country, explore_growth_opportunities, create_advanced_features, prepare_ml_data, train_xgboost_model, forecast_future, forecast_with_arima, forecast_with_sarima, forecast_with_prophet
from backend.comparison import get_channel_summary, get_engagement_stats, get_posting_frequency, vader, save_to_json
from backend.tiktok import tiktok_sentiment_analysis
from backend.twitter import twitter_sentiment_analysis, twitter_comparison, twitter_content_analysis
from backend.ig import ig_sentiment_analysis

app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_TIMEOUT = 3600  # 1 hour in seconds

# Initialize Redis connection
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    redis_client.ping()
    print("Redis connection established")
    USE_REDIS = True
except (redis.ConnectionError, redis.TimeoutError):
    print("Redis not available, falling back to in-memory cache")
    USE_REDIS = False
    cache = {}

def get_cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a cache key based on function name and arguments"""
    key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
    return f"youtube_analytics:{hashlib.md5(key_data.encode()).hexdigest()}"

def get_from_cache(key: str) -> Optional[Any]:
    """Get value from cache (Redis or in-memory)"""
    if USE_REDIS:
        try:
            cached_data = redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Redis get error: {e}")
    else:
        if key in cache:
            result, timestamp = cache[key]
            if time.time() - timestamp < CACHE_TIMEOUT:
                return result
            else:
                del cache[key]
    return None

def set_to_cache(key: str, value: Any, timeout: int = CACHE_TIMEOUT) -> None:
    """Set value to cache (Redis or in-memory)"""
    if USE_REDIS:
        try:
            redis_client.setex(key, timeout, pickle.dumps(value))
        except Exception as e:
            print(f"Redis set error: {e}")
    else:
        cache[key] = (value, time.time())

def clear_cache_all() -> None:
    """Clear all cache entries"""
    if USE_REDIS:
        try:
            keys = redis_client.keys("youtube_analytics:*")
            if keys:
                redis_client.delete(*keys)
        except Exception as e:
            print(f"Redis clear error: {e}")
    else:
        cache.clear()

def cached_result(timeout: int = CACHE_TIMEOUT):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            
            # Check cache first
            result = get_from_cache(cache_key)
            if result is not None:
                print(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            print(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            set_to_cache(cache_key, result, timeout)
            return result
        return wrapper
    return decorator

# Ensure output directory exists
def ensure_output_dir():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def serve_json(filename):
    try:
        output_dir = ensure_output_dir()
        data_file = os.path.join(output_dir, filename)
        with open(data_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})

# Cached data loading functions
@cached_result(timeout=1800)  # Cache for 30 minutes
def load_ishowspeed_data():
    """Load and cache IShowSpeed data"""
    data_file = os.path.join("data", "ishowspeed_latest_500.json")
    if not os.path.exists(data_file):
        return None
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@cached_result(timeout=1800)
def load_creator_data(creator_name):
    """Load and cache creator data"""
    data_file = os.path.join("data", f"{creator_name}_latest_500.json")
    if not os.path.exists(data_file):
        return None
    return load_data_from_file(data_file)

@cached_result(timeout=3600)
def get_monthly_analysis():
    """Cached monthly analysis for IShowSpeed"""
    data = load_ishowspeed_data()
    if not data:
        return None
    
    df, channel_data = load_data(os.path.join("data", "ishowspeed_latest_500.json"))
    
    # Views and likes analysis
    monthly_views, quarterly_views, monthly_growth, quarterly_growth = analyze_views(df)
    monthly_likes, quarterly_likes, monthly_growth_likes = analyze_likes(df)
    
    # Save to JSON files
    save_series_to_json(monthly_views, "monthly_views.json")
    save_series_to_json(monthly_growth, "monthly_growth.json", value_name="growth")
    save_series_to_json(monthly_likes, "monthly_likes.json")
    save_series_to_json(monthly_growth_likes, "monthly_growth_likes.json", value_name="growth")
    
    return True

@cached_result(timeout=3600)
def get_content_analysis():
    """Cached content analysis"""
    data = load_ishowspeed_data()
    if not data:
        return None
    
    return content(data)

@cached_result(timeout=3600)
def get_content_analysis_twitter():
    """Cached content analysis"""
    twitter_content_analysis()
    
    return True

@cached_result(timeout=3600)
def get_sentiment_analysis():
    """Cached sentiment analysis - server optimized version"""
    data = load_ishowspeed_data()
    if not data:
        return None
    
    # Run sentiment analysis without plotting to prevent server hangs
    sentiment_over_time(data, skip_plotting=True)
    sentiment_over_time_monthly(data, skip_plotting=True)
    vader_textblob(data, skip_plotting=True)
    top_positive_negative_comments(data)
    tiktok_sentiment_analysis()
    twitter_sentiment_analysis()
    ig_sentiment_analysis()
    return True

@cached_result(timeout=3600)
def get_predictions():
    """Cached predictions"""
    try:
        data = load_ishowspeed_data()
        if not data:
            return None
        
        forecast_views(data)
        forecast_next_country(data)
        explore_growth_opportunities(data)
        forecast_with_arima(data)
        forecast_with_sarima(data)
        forecast_with_prophet(data)

        df = pd.DataFrame(data['latest_videos'])
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at').reset_index(drop=True)
        print(f"Loaded {len(df)} videos")

        df = create_advanced_features(df)
        X, y, df_clean, feature_cols = prepare_ml_data(df)
        xgb_model, metrics = train_xgboost_model(X, y)
        forecast_future(df_clean, xgb_model, feature_cols, days=180)

        return True
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        # Clear cache for this function to prevent repeated failures
        cache_key = get_cache_key('get_predictions')
        if USE_REDIS:
            try:
                redis_client.delete(cache_key)
            except:
                pass
        else:
            cache.pop(cache_key, None)
        raise e

@cached_result(timeout=3600)
def get_creator_comparisons():
    """Cached creator comparisons"""
    creators = ["ishowspeed", "mrbeast", "kaicenat", "pewdiepie"]
    
    for creator in creators:
        creator_data = load_creator_data(creator)
        if creator_data:
            get_channel_summary(creator_data, f"{creator}_summary.json")
            get_engagement_stats(creator_data, f"{creator}_engagement.json")
            get_posting_frequency(creator_data, f"{creator}_posting_freq.json")
            vader(creator_data, f"{creator}_vader.json")
            
            # Monthly views for other creators
            if creator != "ishowspeed":
                df, channel_data = load_data_from_file_df(os.path.join("data", f'{creator}_latest_500.json'))
                monthly_views, _, _, _ = analyze_views(df)
                save_series_to_json(monthly_views, f"{creator}_monthly_views.json")

    twitter_comparison()
    return True

# API Routes with lazy loading
@app.route('/api/channel_data')
def api_channel_data():
    try:
        data_file = os.path.join("data", "ishowspeed_latest_500.json")
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

# Routes that trigger analysis on demand
@app.route('/api/monthly_views')
def api_monthly_views():
    get_monthly_analysis()
    return serve_json("monthly_views.json")

@app.route('/api/monthly_growth')
def api_monthly_growth():
    get_monthly_analysis()
    return serve_json("monthly_growth.json")

@app.route('/api/monthly_likes')
def api_monthly_likes():
    get_monthly_analysis()
    return serve_json("monthly_likes.json")

@app.route('/api/monthly_growth_likes')
def api_monthly_growth_likes():
    get_monthly_analysis()
    return serve_json("monthly_growth_likes.json")

@app.route('/api/content_engagement')
def api_content_engagement():
    get_content_analysis()
    return serve_json("content_engagement.json")

@app.route('/api/content_sentiment')
def api_content_sentiment():
    get_content_analysis()
    return serve_json("content_sentiment.json")

@app.route('/api/content_sentiment_percentage')
def api_content_sentiment_percentage():
    get_content_analysis()
    return serve_json("content_sentiment_percentage.json")

@app.route('/api/twitter_ishowspeed_content_results')
def api_twitter_ishowspeed_content_results():
    get_content_analysis_twitter()
    return serve_json("twitter_ishowspeed_content_results.json")

@app.route('/api/twitter_mrbeast_content_results')
def api_twitter_mrbeast_content_results():
    get_content_analysis_twitter()
    return serve_json("twitter_mrbeast_content_results.json")

@app.route('/api/sentiment_summary')
def api_sentiment_summary():
    get_sentiment_analysis()
    return serve_json("sentiment_summary.json")

@app.route('/api/smoothed_sentiment_trend')
def api_smoothed_sentiment_trend():
    get_sentiment_analysis()
    return serve_json("smoothed_sentiment_trend.json")

@app.route('/api/monthly_sentiment_trend')
def api_monthly_sentiment_trend():
    get_sentiment_analysis()
    return serve_json("monthly_sentiment_trend.json")

@app.route('/api/top_3_videos')
def api_top_3_videos():
    get_sentiment_analysis()
    return serve_json("top_3_videos.json")

@app.route('/api/top_sentiment_comments')
def api_top_sentiment_comments():
    get_sentiment_analysis()
    return serve_json("top_sentiment_comments.json")

@app.route('/api/tiktok_sentiment_results')
def api_tiktok_sentiment_results():
    get_sentiment_analysis()
    return serve_json("tiktok_sentiment_results.json")

@app.route('/api/twitter_sentiment_results')
def api_twitter_sentiment_results():
    get_sentiment_analysis()
    return serve_json("twitter_sentiment_results.json")

@app.route('/api/ig_sentiment_results')
def api_ig_sentiment_results():
    get_sentiment_analysis()
    return serve_json("ig_sentiment_results.json")

@app.route('/api/forecasted_views')
def api_forecasted_views():
    get_predictions()
    return serve_json("forecasted_views_combined.json")

@app.route('/api/predicted_countries')
def api_predicted_countries():
    get_predictions()
    return serve_json("predicted_countries.json")

@app.route('/api/explore_growth_opportunities')
def api_explore_growth_opportunities():
    get_predictions()
    return serve_json("explore_growth_opportunities.json")

@app.route('/api/combined_6mo_forecast')
def api_combined_6mo_forecast():
    get_predictions()
    return serve_json("combined_6mo_forecast.json")

@app.route('/api/forecasted_views_arima')
def api_forecasted_views_arima():
    get_predictions()
    return serve_json("forecasted_views_arima.json")

@app.route('/api/forecasted_views_sarima')
def api_forecasted_views_sarima():
    get_predictions()
    return serve_json("forecasted_views_sarima.json")

@app.route('/api/forecasted_views_prophet')
def api_forecasted_views_prophet():
    get_predictions()
    return serve_json("forecasted_views_prophet.json")

# Creator comparison routes
@app.route('/api/ishowspeed_summary')
def api_ishowspeed_summary():
    get_creator_comparisons()
    return serve_json("ishowspeed_summary.json")

@app.route('/api/mrbeast_summary')
def api_mrbeast_summary():
    get_creator_comparisons()
    return serve_json("mrbeast_summary.json")

@app.route('/api/kaicenat_summary')
def api_kaicenat_summary():
    get_creator_comparisons()
    return serve_json("kaicenat_summary.json")

@app.route('/api/pewdiepie_summary')
def api_pewdiepie_summary():
    get_creator_comparisons()
    return serve_json("pewdiepie_summary.json")

@app.route('/api/ishowspeed_engagement')
def api_ishowspeed_engagement():
    get_creator_comparisons()
    return serve_json("ishowspeed_engagement.json")

@app.route('/api/mrbeast_engagement')
def api_mrbeast_engagement():
    get_creator_comparisons()
    return serve_json("mrbeast_engagement.json")

@app.route('/api/kaicenat_engagement')
def api_kaicenat_engagement():
    get_creator_comparisons()
    return serve_json("kaicenat_engagement.json")

@app.route('/api/pewdiepie_engagement')
def api_pewdiepie_engagement():
    get_creator_comparisons()
    return serve_json("pewdiepie_engagement.json")

@app.route('/api/ishowspeed_posting_freq')
def api_ishowspeed_posting_freq():
    get_creator_comparisons()
    return serve_json("ishowspeed_posting_freq.json")

@app.route('/api/mrbeast_posting_freq')
def api_mrbeast_posting_freq():
    get_creator_comparisons()
    return serve_json("mrbeast_posting_freq.json")

@app.route('/api/kaicenat_posting_freq')
def api_kaicenat_posting_freq():
    get_creator_comparisons()
    return serve_json("kaicenat_posting_freq.json")

@app.route('/api/pewdiepie_posting_freq')
def api_pewdiepie_posting_freq():
    get_creator_comparisons()
    return serve_json("pewdiepie_posting_freq.json")

@app.route('/api/ishowspeed_vader')
def api_ishowspeed_vader():
    get_creator_comparisons()
    return serve_json("ishowspeed_vader.json")

@app.route('/api/mrbeast_vader')
def api_mrbeast_vader():
    get_creator_comparisons()
    return serve_json("mrbeast_vader.json")

@app.route('/api/kaicenat_vader')
def api_kaicenat_vader():
    get_creator_comparisons()
    return serve_json("kaicenat_vader.json")

@app.route('/api/pewdiepie_vader')
def api_pewdiepie_vader():
    get_creator_comparisons()
    return serve_json("pewdiepie_vader.json")

@app.route('/api/mrbeast_monthly_views')
def api_mrbeast_monthly_views():
    get_creator_comparisons()
    return serve_json("mrbeast_monthly_views.json")

@app.route('/api/kaicenat_monthly_views')
def api_kaicenat_monthly_views():
    get_creator_comparisons()
    return serve_json("kaicenat_monthly_views.json")

@app.route('/api/pewdiepie_monthly_views')
def api_pewdiepie_monthly_views():
    get_creator_comparisons()
    return serve_json("pewdiepie_monthly_views.json")

@app.route('/api/total_engagement_rate')
def api_total_engagement_rate():
    get_creator_comparisons()
    return serve_json("total_engagement_rate.json")

@app.route('/api/twitter_avg_engagement')
def api_twitter_avg_engagement():
    get_creator_comparisons()
    return serve_json("twitter_avg_engagement.json")

# Cache management routes
@app.route('/api/cache/clear')
def clear_cache():
    """Clear all cached data"""
    clear_cache_all()
    return jsonify({"message": "Cache cleared successfully"})

@app.route('/api/cache/clear_sentiment')
def clear_sentiment_cache():
    """Clear only sentiment analysis cache"""
    cache_key = get_cache_key('get_sentiment_analysis')
    if USE_REDIS:
        try:
            redis_client.delete(cache_key)
        except Exception as e:
            return jsonify({"error": f"Redis error: {e}"})
    else:
        cache.pop(cache_key, None)
    return jsonify({"message": "Sentiment analysis cache cleared successfully"})

@app.route('/api/cache/status')
def cache_status():
    """Get cache status"""
    if USE_REDIS:
        try:
            keys = redis_client.keys("youtube_analytics:*")
            cache_info = {}
            for key in keys:
                ttl = redis_client.ttl(key)
                cache_info[key.decode()] = {
                    "ttl_seconds": ttl,
                    "expires_in": ttl if ttl > 0 else 0
                }
            return jsonify({
                "cache_type": "Redis",
                "total_keys": len(keys),
                "keys": cache_info
            })
        except Exception as e:
            return jsonify({"error": f"Redis error: {e}"})
    else:
        cache_info = {}
        for key, (_, timestamp) in cache.items():
            age = time.time() - timestamp
            cache_info[key] = {
                "age_seconds": int(age),
                "expires_in": int(CACHE_TIMEOUT - age) if age < CACHE_TIMEOUT else 0
            }
        return jsonify({
            "cache_type": "In-Memory",
            "total_keys": len(cache),
            "keys": cache_info
        })

# Static file routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

def main():
    """Fast startup - no data processing on startup"""
    load_dotenv()
    ensure_output_dir()
    
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('FLASK_PORT', os.environ.get('PORT', 5000)))
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    cache_type = "Redis" if USE_REDIS else "In-Memory"
    print(f"Starting Flask server with {cache_type} caching enabled...")
    print("Data will be processed on-demand when endpoints are accessed.")
    print("Access /api/cache/status to see cache information.")
    print("Access /api/cache/clear to clear the cache.")
    print(f"Server starting on {host}:{port}")
    
    app.run(debug=debug, port=port, host=host)

if __name__ == "__main__":
    main()