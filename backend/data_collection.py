import json
import requests
from bs4 import BeautifulSoup
import re
from googleapiclient.discovery import build
from time import sleep
import os
from dotenv import load_dotenv

# Define the channels with their IDs
CREATORS = {
    "ishowspeed": {
        "channel_id": "UCWsDFcIhY2DBi3GB5uykGXA"
    },
    "mrbeast": {
        "channel_id": "UCX6OQ3DkcsbYNE6H8uQQuVA"
    },
    "kaicenat": {
        "channel_id": "UCvCfpQXRXdJdL07pzTIA6Cw"
    },
    "pewdiepie": {
        "channel_id": "UC-lHJZR3Gqxm24_Vd_AJ5Yw"
    }
}

def get_channel_info():
    # """Get basic information about IShowSpeed from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/IShowSpeed'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract all paragraph text from the main content
    paragraphs = soup.select('div.mw-parser-output > p')
    content = ''
    for p in paragraphs[:5]:
        text = p.get_text()  # Get the first 5 paragraphs
        content += re.sub(r'\[\d+\]', '', text)
    
    return content

def collect_youtube_data(api_key, channel_id, max_videos=200):
    # """Collect YouTube data for the specified channel"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Step 1: Get Uploads playlist ID
    channel_response = youtube.channels().list(
        part='contentDetails,snippet,statistics',
        id=channel_id
    ).execute()
    
    channel = channel_response['items'][0]
    uploads_playlist_id = channel['contentDetails']['relatedPlaylists']['uploads']
    channel_name = channel['snippet']['title']
    channel_stats = channel['statistics']
    
    channel_data = {
        "channel_name": channel_name,
        "subscriber_count": int(channel_stats.get("subscriberCount", 0)),
        "total_view_count": int(channel_stats.get("viewCount", 0)),
        "video_count": int(channel_stats.get("videoCount", 0)),
    }
    
    # Step 2: Fetch videos 
    videos = []
    next_page_token = None
    
    while len(videos) < max_videos:
        playlist_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
    
        for item in playlist_response['items']:
            video_id = item['contentDetails']['videoId']
    
            # Fetch video details
            video_response = youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()
    
            if not video_response['items']:
                continue
    
            video = video_response['items'][0]
            snippet = video['snippet']
            stats = video['statistics']
    
            video_data = {
                "video_id": video_id,
                "title": snippet['title'],
                "published_at": snippet['publishedAt'],
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0))
            }
    
            videos.append(video_data)
    
            if len(videos) >= max_videos:
                break
    
        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break
    
    return channel_data, videos

def collect_video_comments(youtube, videos, max_comments=100):
    # """Collect comments for each video"""
    for video in videos:
        video_id = video["video_id"]
        comments = []
        next_comment_page = None
    
        while len(comments) < max_comments:
            try:
                comment_response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_comment_page,
                    textFormat="plainText"
                ).execute()
    
                for item in comment_response["items"]:
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(top_comment)
    
                    if len(comments) >= max_comments:
                        break
    
                next_comment_page = comment_response.get("nextPageToken")
                if not next_comment_page:
                    break
    
            except Exception as e:
                print(f"Failed to get comments for video {video_id}: {e}")
                break

            # Sleep for 1 second to avoid hitting the API rate limit
            sleep(1)
    
        # Add comments to the video entry
        video["comments"] = comments
    
    return videos

def save_data(channel_data, videos, filename="ishowspeed_latest_200.json"):
    """Save the collected data to a JSON file"""
    output = {
        "channel_data": channel_data,
        "latest_videos": videos
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(videos)} videos (with comments) for {channel_data['channel_name']} in '{filename}'")
    except Exception as e:
        print(f"Failed to save data: {e}")

    return output

def load_data(filename="data/ishowspeed_latest_500.json"):
    """Load data from a JSON file"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_data_from_file(filename):
    """Load data from a JSON file"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def collect_for_creator(api_key, name, channel_id, max_videos=500, max_comments=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    
    print(f"\nCollecting data for {name.title()}...")
    channel_data, videos = collect_youtube_data(api_key, channel_id, max_videos=max_videos)
    if name == "ishowspeed":
        channel_description = get_channel_info()
        channel_data["description"] = channel_description 
    videos = collect_video_comments(youtube, videos, max_comments=max_comments)
    save_data(channel_data, videos, filename=f"{name}_latest_{max_videos}.json")

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")

    if not api_key:
        print("Missing API_KEY in .env file")
        return

    for name, details in CREATORS.items():
        collect_for_creator(api_key, name, details["channel_id"])

if __name__ == "__main__":
    main()