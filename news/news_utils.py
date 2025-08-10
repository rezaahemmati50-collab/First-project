# news/news_utils.py
import os
import csv
import requests
import pandas as pd
import json

# Try to read config/settings.json for API key
def _load_config():
    cfg_path = os.path.join("config", "settings.json")
    sample_path = os.path.join("config", "settings.example.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    elif os.path.exists(sample_path):
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return cfg

def get_news_from_csv(path, limit=10):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # normalize columns
        cols = [c.strip().lower() for c in df.columns]
        # expect columns: date,title,source,url
        rows = []
        for idx, r in df.iterrows():
            title = r.get('title') if 'title' in r else (r.get('Title') if 'Title' in r else None)
            date = r.get('date') if 'date' in r else None
            source = r.get('source') if 'source' in r else None
            url = r.get('url') if 'url' in r else None
            rows.append({'date': date, 'title': title, 'source': source, 'url': url})
        return rows[:limit]
    except Exception:
        return []

def get_news_from_newsapi(api_key, q="crypto OR bitcoin OR ethereum", page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        j = resp.json()
        items = []
        for art in j.get("articles", []):
            items.append({
                "date": art.get("publishedAt"),
                "title": art.get("title"),
                "source": art.get("source", {}).get("name"),
                "url": art.get("url")
            })
        return items
    except Exception:
        return []

def fetch_news(limit=10):
    cfg = _load_config()
    api_key = cfg.get("newsapi_key", "") if cfg else ""
    if api_key:
        items = get_news_from_newsapi(api_key, page_size=limit)
        if items:
            return items
    # fallback to local CSV
    sample_path = os.path.join("data", "sample_news.csv")
    return get_news_from_csv(sample_path, limit=limit)
