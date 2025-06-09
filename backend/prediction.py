import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using LinearRegression as fallback.")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from langdetect import detect, LangDetectException
from collections import Counter
import pycountry
import json
import seaborn as sns
import warnings

# Suppress pandas timezone warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
import os
import hashlib
import requests
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def prepare_data(data):
    """Helper function to prepare data for all models"""
    df = pd.DataFrame(data['latest_videos'])
    df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
    df = df.sort_values('published_at')
    df['cumulative_views'] = df['view_count'].cumsum()
    
    # For time series models we need regular frequency - resample to daily
    ts_df = df.set_index('published_at')['view_count'].resample('D').sum().cumsum().reset_index()
    ts_df.columns = ['ds', 'y']
    
    return df, ts_df

def save_forecast(actual_df, future_dates, forecast_values, model_name):
    """Helper function to save forecast results with last 6 months of actual data"""
    # Make a copy of the actual data
    actual_df = actual_df.copy()
    
    # Convert dates if needed and filter last 6 months
    if not pd.api.types.is_datetime64_any_dtype(actual_df['ds']):
        actual_df['ds'] = pd.to_datetime(actual_df['ds'])
    
    # Get the cutoff date (6 months before the last date)
    cutoff_date = actual_df['ds'].max() - pd.DateOffset(months=6)
    
    # Filter to only keep last 6 months of data
    last_six_months = actual_df[actual_df['ds'] >= cutoff_date]
    
    # Prepare actual data list
    actual_list = last_six_months.copy()
    actual_list['ds'] = actual_list['ds'].dt.strftime('%Y-%m-%d')
    actual_list = actual_list[['ds', 'y']].rename(columns={'ds': 'date', 'y': 'views'}).to_dict(orient='records')
    
    # Prepare forecast data list
    forecast_list = [
        {'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d, 
         'views': int(v)}
        for d, v in zip(future_dates, forecast_values)
    ]
    
    combined_output = {
        'actual': actual_list,
        'forecast': forecast_list
    }
    
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    combined_path = os.path.join(output_dir, f"forecasted_views_{model_name}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_output, f, indent=4)
    
    print(f"Saved {model_name} forecast to {combined_path}")
    return combined_path

def forecast_with_arima(data):
    """Forecast using ARIMA model"""
    df, ts_df = prepare_data(data)
    
    # Fit ARIMA model
    model = ARIMA(ts_df['y'], order=(5,1,0))  # (p,d,q) parameters
    model_fit = model.fit()
    
    # Forecast next 180 days
    forecast = model_fit.forecast(steps=180)
    future_dates = pd.date_range(ts_df['ds'].max(), periods=181, freq='D')[1:]
    
    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(ts_df['ds'], ts_df['y'], label="Actual Views")
    plt.plot(future_dates, forecast, label="ARIMA Forecast", linestyle='--')
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    save_forecast(ts_df, future_dates, forecast, "arima")
    return forecast

def forecast_with_sarima(data):
    """Forecast using SARIMA model (with seasonality)"""
    df, ts_df = prepare_data(data)
    
    # Fit SARIMA model (weekly seasonality)
    model = SARIMAX(ts_df['y'], order=(1,1,1), seasonal_order=(1,1,1,7))
    model_fit = model.fit()
    
    # Forecast next 180 days
    forecast = model_fit.forecast(steps=180)
    future_dates = pd.date_range(ts_df['ds'].max(), periods=181, freq='D')[1:]
    
    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(ts_df['ds'], ts_df['y'], label="Actual Views")
    plt.plot(future_dates, forecast, label="SARIMA Forecast", linestyle='--')
    plt.title("SARIMA Forecast (with Seasonality)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    save_forecast(ts_df, future_dates, forecast, "sarima")
    return forecast

def forecast_with_prophet(data):
    """Forecast using Facebook Prophet"""
    df, ts_df = prepare_data(data)
    
    # Remove timezone information if present
    if pd.api.types.is_datetime64tz_dtype(ts_df['ds']):
        ts_df['ds'] = ts_df['ds'].dt.tz_localize(None)
    
    # Fit Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(ts_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=180, freq='D')
    forecast = model.predict(future)
    
    # Filter only future dates
    future_forecast = forecast[forecast['ds'] > ts_df['ds'].max()]
    
    # Create and save plot without showing
    fig = model.plot(forecast)
    plt.title("Prophet Forecast")
    
    # Save plot to file
    plot_dir = os.path.join(os.getcwd(), "output", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "prophet_forecast.png")
    # plt.savefig(plot_path)
    plt.close()
    
    # Save forecast data
    json_path = save_forecast(ts_df, future_forecast['ds'], future_forecast['yhat'], "prophet")
    
    return {
        'plot_path': plot_path,
        'json_path': json_path,
        'forecast': future_forecast['yhat'].values.tolist()
    }

def forecast_views(data):
    """Forecast view count for the next few days using linear regression"""
    # Load video data
    df = pd.DataFrame(data['latest_videos'])

    # Parse datetime
    df['published_at'] = pd.to_datetime(df['published_at'])

    # Sort by time
    df = df.sort_values('published_at')

    # Convert dates to ordinal (for regression)
    df['days_since_start'] = (df['published_at'] - df['published_at'].min()).dt.days

    # Aggregate views over time
    df['cumulative_views'] = df['view_count'].cumsum()

    # Linear Regression model for forecasting views
    X = df[['days_since_start']]
    y = df['cumulative_views']

    model = LinearRegression()
    model.fit(X, y)

    # Predict for next 180 days (6 months)
    future_days = np.arange(df['days_since_start'].max() + 1, df['days_since_start'].max() + 181).reshape(-1, 1)
    future_views = model.predict(future_days)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(df['published_at'], df['cumulative_views'], label="Actual Views", color='blue')
    future_dates = [df['published_at'].min() + timedelta(days=int(x)) for x in future_days.flatten()]
    plt.plot(future_dates, future_views, label="Forecasted Views", linestyle='--', color='green')
    plt.title("Forecast: IShowSpeed's Cumulative Views")
    plt.xlabel("Date")
    plt.ylabel("Total Views")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # Predicted views after 6 months
    predicted_total_views = int(future_views[-1])
    print(f"Projected total views in 6 months: {predicted_total_views:,}")

    # Prepare output data
    forecast_df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'predicted_views': future_views.astype(int)
    })

    # Save to combined JSON
    actual_data = df[['published_at', 'cumulative_views']].copy()
    actual_data['published_at'] = actual_data['published_at'].dt.strftime('%Y-%m-%d')
    actual_list = actual_data.rename(columns={
        'published_at': 'date',
        'cumulative_views': 'views'
    }).to_dict(orient='records')

    forecast_list = [
        {'date': d.strftime('%Y-%m-%d'), 'views': int(v)}
        for d, v in zip(future_dates, future_views)
    ]

    combined_output = {
        'actual': actual_list,
        'forecast': forecast_list
    }

    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save combined JSON
    combined_path = os.path.join(output_dir, "forecasted_views_combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_output, f, indent=4)

    print(f"Combined actual and forecasted views saved to {combined_path}")

def forecast_next_country(data):
    df = pd.DataFrame(data["latest_videos"])

    # Flatten all comments
    all_comments = sum((video.get("comments", []) for video in df.to_dict("records")), [])

    # Detect language with error handling
    languages = []
    for c in all_comments:
        c = c.strip()
        if len(c) > 10:
            try:
                lang = detect(c)
                languages.append(lang)
            except LangDetectException:
                continue  # Skip undetectable comments

    # Count language codes
    lang_counts = Counter(languages)

    # Convert language codes to full names
    full_lang_counts = {}
    for code, count in lang_counts.items():
        try:
            lang_name = pycountry.languages.get(alpha_2=code).name
        except:
            lang_name = code  # fallback to code if not found
        full_lang_counts[lang_name] = count

    # Display nicely
    # for lang, count in full_lang_counts.items():
    #     print(f"{lang}: {count}")

        # Countries he's already visited
    with open("data/ishowspeed_countries.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    visited = data

    # Language-to-country mapping (simplified)
    lang_country_map = {
        'en': ['United States', 'United Kingdom', 'Canada', 'Australia'],
        'so': ['Somalia'],
        'id': ['Indonesia'],
        'de': ['Germany', 'Austria', 'Switzerland'],
        'es': ['Spain', 'Mexico', 'Argentina', 'Colombia'],
        'tl': ['Philippines'],
        'af': ['South Africa'],
        'nl': ['Netherlands', 'Belgium'],
        'it': ['Italy'],
        'pl': ['Poland'],
        'cy': ['Wales'],
        'et': ['Estonia'],
        'fr': ['France', 'Belgium', 'Canada'],
        'pt': ['Portugal', 'Brazil'],
        'no': ['Norway'],
        'da': ['Denmark'],
        'ro': ['Romania'],
        'fi': ['Finland'],
        'sw': ['Kenya', 'Tanzania'],
        'tr': ['Turkey'],
        'vi': ['Vietnam'],
        'ca': ['Spain'],
        'sv': ['Sweden'],
        'sk': ['Slovakia'],
        'sl': ['Slovenia'],
        'ar': ['Saudi Arabia', 'UAE', 'Egypt'],
        'cs': ['Czech Republic'],
        'hr': ['Croatia'],
        'hu': ['Hungary'],
        'zh-cn': ['China'],
        'zh-tw': ['Taiwan'],
        'ko': ['South Korea'],
        'ru': ['Russia'],
        'sq': ['Albania'],
        'th': ['Thailand'],
        'lt': ['Lithuania'],
        'bg': ['Bulgaria'],
        'lv': ['Latvia'],
        'ja': ['Japan'],
        'el': ['Greece'],
        'fa': ['Iran'],
        'uk': ['Ukraine'],
        'ta': ['India', 'Sri Lanka'],
        'hi': ['India'],
        'mk': ['North Macedonia'],
        'he': ['Israel'],
        'te': ['India'],
        'ur': ['Pakistan']
    }

    # Rank candidate countries
    candidate_scores = {}

    for lang, count in lang_counts.items():
        countries = lang_country_map.get(lang, [])
        for country in countries:
            if country not in visited:
                candidate_scores[country] = candidate_scores.get(country, 0) + count

    # Sort by popularity score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    # Display top 10 predictions
    print("Predicted countries IShowSpeed might visit next:\n")
    for i, (country, score) in enumerate(sorted_candidates[:10], 1):
        print(f"{i}. {country} (language score: {score})")

    # Visualization of top 10 predicted countries
    top_countries = sorted_candidates[:10]
    countries, scores = zip(*top_countries)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(scores), y=list(countries), palette='viridis')
    plt.title("Top 10 Predicted Countries IShowSpeed Might Visit Next")
    plt.xlabel("Language-Based Popularity Score")
    plt.ylabel("Country")
    plt.tight_layout()
    # plt.show()

    # Save top 10 predicted countries to JSON
    top_country_predictions = [{"country": country, "score": score} for country, score in top_countries]

    # Ensure output folder exists
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    country_json_path = os.path.join(output_dir, "predicted_countries.json")
    with open(country_json_path, "w", encoding="utf-8") as f:
        json.dump(top_country_predictions, f, indent=4)

    print(f"Predicted countries saved to {country_json_path}")

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

def build_prompt(latest_videos):
    # Use top 5 videos by view count
    top_videos = sorted(latest_videos, key=lambda v: v["view_count"], reverse=True)[:5]

    prompt = "Based on the following popular videos by IShowSpeed, suggest 4-5 creative content ideas that match his energetic, international IRL and football-themed style:\n\n"
    for video in top_videos:
        prompt += f"Title: {video['title']}\n"
        prompt += f"Views: {video['view_count']:,}\n"
        if "comments" in video and video["comments"]:
            prompt += f"Top Comment: {video['comments'][0]}\n"
        prompt += "\n"

    prompt += "Suggest the next 4-5 video ideas:"
    return prompt

def get_cached_response(prompt):
    cache_file = os.path.join(output_dir, "explore_growth_opportunities.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
            if cached["prompt"] == prompt:
                print("Using cached recommendation.")
                return cached
    return None

def save_to_cache(prompt, response_text):
    cache_file = os.path.join(output_dir, "explore_growth_opportunities.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": response_text}, f, indent=4)
    print(f"Saved to cache: {cache_file}")


def query_openrouter(prompt):
    print("Querying OpenRouter API...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3.7-sonnet",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")

def recommend_content(data):
    latest_videos = data.get("latest_videos", [])
    if not latest_videos:
        print("No videos found.")
        return

    prompt = build_prompt(latest_videos)

    # Check cache
    cached = get_cached_response(prompt)
    if cached:
        print("\n--- Recommended Content Ideas (from cache) ---\n")
        print(cached["response"])
        return

    # If not cached, call API
    response = query_openrouter(prompt)
    print("\n--- Recommended Content Ideas ---\n")
    print(response)

    # Save to cache
    save_to_cache(prompt, response)

def explore_growth_opportunities(data):
    latest_videos = data.get("latest_videos", [])
    channel_info = data.get("channel_data", {})
    
    prompt = f"""
Based on the content below from IShowSpeed's channel, analyze and suggest potential areas of growth for him. Consider things like:

- New platforms (e.g., Kick, Tiktok, YouTube Shorts, etc.)
- New audiences (age groups, countries, subcultures)
- New video formats (e.g., documentaries, skits, challenges)
- Collaborations (with creators, brands, athletes)
- Ways to deepen brand identity or expand influence

Channel Info:
Name: {channel_info.get("channel_name", "N/A")}
Subscribers: {channel_info.get("subscriber_count", 0):,}
Total Views: {channel_info.get("total_view_count", 0):,}
Video Count: {channel_info.get("video_count", 0)}
Description: {channel_info.get("description", "")[:500]}...

Recent Content:
"""

    for video in latest_videos[:5]:
        prompt += f"- {video['title']} ({video['view_count']:,} views)\n"

    prompt += "\nWhat growth strategies or new directions would you recommend for IShowSpeed?"

    # Check cache
    cached = get_cached_response(prompt)
    if cached:
        print("\n--- Growth Opportunities (from cache) ---\n")
        print(cached["response"])
        return

    # Call OpenRouter
    response = query_openrouter(prompt)
    print("\n--- Growth Opportunities ---\n")
    print(response)

    # Save
    save_to_cache(prompt, response)

np.random.seed(42)

def create_advanced_features(df):
    print("Creating advanced features...")
    df['hour'] = df['published_at'].dt.hour
    df['day_of_week'] = df['published_at'].dt.dayofweek
    df['day_of_month'] = df['published_at'].dt.day
    df['month'] = df['published_at'].dt.month
    df['quarter'] = df['published_at'].dt.quarter
    df['year'] = df['published_at'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['title_caps_ratio'] = df['title'].str.count(r'[A-Z]') / df['title_length']
    df['title_exclamation_count'] = df['title'].str.count('!')
    df['title_question_count'] = df['title'].str.count(r'\?')

    df['like_to_view_ratio'] = df['like_count'] / np.maximum(df['view_count'], 1)
    df['comment_to_view_ratio'] = df['comment_count'] / np.maximum(df['view_count'], 1)
    df['engagement_score'] = (df['like_count'] + df['comment_count']) / np.maximum(df['view_count'], 1)

    for window in [3, 7, 14, 30]:
        df[f'view_count_rolling_mean_{window}'] = df['view_count'].rolling(window=window, min_periods=1).mean()
        df[f'view_count_rolling_std_{window}'] = df['view_count'].rolling(window=window, min_periods=1).std()
        df[f'engagement_rolling_mean_{window}'] = df['engagement_score'].rolling(window=window, min_periods=1).mean()

    for lag in [1, 2, 3, 7]:
        df[f'view_count_lag_{lag}'] = df['view_count'].shift(lag)
        df[f'engagement_lag_{lag}'] = df['engagement_score'].shift(lag)

    df['days_since_start'] = (df['published_at'] - df['published_at'].min()).dt.days
    df['video_frequency'] = df.index + 1
    df['days_between_videos'] = df['published_at'].diff().dt.days.fillna(0)

    df['is_prime_time'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 10)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] <= 17)).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

def prepare_ml_data(df, target_col='view_count'):
    print("Preparing ML data...")
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'day_of_month', 'quarter', 'is_weekend', 'is_prime_time', 'is_morning', 'is_afternoon',
        'title_length', 'title_word_count', 'title_caps_ratio', 'title_exclamation_count', 'title_question_count',
        'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score',
        'view_count_rolling_mean_3', 'view_count_rolling_mean_7', 'view_count_rolling_mean_14', 'view_count_rolling_mean_30',
        'view_count_rolling_std_3', 'view_count_rolling_std_7', 'view_count_rolling_std_14', 'view_count_rolling_std_30',
        'engagement_rolling_mean_3', 'engagement_rolling_mean_7', 'engagement_rolling_mean_14', 'engagement_rolling_mean_30',
        'view_count_lag_1', 'view_count_lag_2', 'view_count_lag_3', 'view_count_lag_7',
        'engagement_lag_1', 'engagement_lag_2', 'engagement_lag_3', 'engagement_lag_7',
        'days_since_start', 'video_frequency', 'days_between_videos'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].copy()
    y = df['view_count'].copy()
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    return X[valid_idx], y[valid_idx], df[valid_idx].copy(), feature_cols

def train_xgboost_model(X, y):
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if XGBOOST_AVAILABLE:
        print("\n=== Training XGBoost Model ===")
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    else:
        print("\n=== Training Linear Regression Model (XGBoost fallback) ===")
        model = LinearRegression()
    
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Train MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}, R²: {test_r2:.4f}")

    return model, {'train_mae': train_mae, 'test_mae': test_mae, 'train_r2': train_r2, 'test_r2': test_r2, 'y_test': y_test.tolist(), 'y_pred_test': y_pred_test.tolist()}

def forecast_future(df, model, feature_cols, days=180):
    print("\nGenerating 6-month forecast...")
    last_date = df['published_at'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')

    future_df = pd.DataFrame({'published_at': future_dates})
    future_df['hour'] = 20
    future_df['day_of_week'] = future_df['published_at'].dt.dayofweek
    future_df['day_of_month'] = future_df['published_at'].dt.day
    future_df['month'] = future_df['published_at'].dt.month
    future_df['quarter'] = future_df['published_at'].dt.quarter
    future_df['year'] = future_df['published_at'].dt.year
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)

    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
    future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

    future_df['title_length'] = df['title_length'].mean()
    future_df['title_word_count'] = df['title_word_count'].mean()
    future_df['title_caps_ratio'] = df['title_caps_ratio'].mean()
    future_df['title_exclamation_count'] = df['title_exclamation_count'].mean()
    future_df['title_question_count'] = df['title_question_count'].mean()
    future_df['like_to_view_ratio'] = df['like_to_view_ratio'].mean()
    future_df['comment_to_view_ratio'] = df['comment_to_view_ratio'].mean()
    future_df['engagement_score'] = df['engagement_score'].mean()

    for window in [3, 7, 14, 30]:
        future_df[f'view_count_rolling_mean_{window}'] = df[f'view_count_rolling_mean_{window}'].mean()
        future_df[f'view_count_rolling_std_{window}'] = df[f'view_count_rolling_std_{window}'].mean()
        future_df[f'engagement_rolling_mean_{window}'] = df[f'engagement_rolling_mean_{window}'].mean()

    for lag in [1, 2, 3, 7]:
        future_df[f'view_count_lag_{lag}'] = df[f'view_count_lag_{lag}'].mean()
        future_df[f'engagement_lag_{lag}'] = df[f'engagement_lag_{lag}'].mean()

    future_df['days_since_start'] = df['days_since_start'].max() + np.arange(1, days + 1)
    future_df['video_frequency'] = df['video_frequency'].max() + np.arange(1, days + 1)
    future_df['days_between_videos'] = df['days_between_videos'].mean()
    future_df['is_prime_time'] = ((future_df['hour'] >= 18) & (future_df['hour'] <= 22)).astype(int)
    future_df['is_morning'] = ((future_df['hour'] >= 6) & (future_df['hour'] <= 10)).astype(int)
    future_df['is_afternoon'] = ((future_df['hour'] >= 12) & (future_df['hour'] <= 17)).astype(int)

    X_future = future_df[feature_cols]
    y_future_pred = model.predict(X_future)

    forecast = pd.DataFrame({
        'date': future_df['published_at'],
        'predicted_views': y_future_pred.astype(int)
    })

    # Include past 6 months historical data
    six_months_ago = df['published_at'].max() - pd.DateOffset(months=6)
    historical = df[df['published_at'] >= six_months_ago][['published_at', 'view_count']].copy()
    historical = historical.rename(columns={'published_at': 'date', 'view_count': 'predicted_views'})
    historical['type'] = 'historical'
    forecast['type'] = 'forecast'

    full_data = pd.concat([historical, forecast], ignore_index=True)
    full_data['date'] = full_data['date'].dt.strftime('%Y-%m-%d')

    full_data.to_json('output/combined_6mo_forecast.json', orient='records', indent=4)
    print("Saved 6-month combined forecast to output/combined_6mo_forecast.json")

def main():
    from data_collection import load_data
    data = load_data()
    # forecast_views(data)
    # forecast_next_country(data)
    # recommend_content(data)
    # explore_growth_opportunities(data)

    forecast_with_arima(data)
    forecast_with_sarima(data)
    forecast_with_prophet(data)

if __name__ == "__main__":
    main()