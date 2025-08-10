# news/news_utils.py
import os
import json
import requests
import pandas as pd

def _load_config():
    cfg = {}
    cfg_path = os.path.join("config", "settings.json")
    example_path = os.path.join("config", "settings.example.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        elif os.path.exists(example_path):
            with open(example_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
    except Exception:
        cfg = {}
    return cfg

def get_news_from_csv(path, limit=10):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # map columns to expected keys
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date": r.get('date') if 'date' in r else r.get('Date'),
                "title": r.get('title') if 'title' in r else r.get('Title'),
                "source": r.get('source') if 'source' in r else r.get('Source'),
                "url": r.get('url') if 'url' in r else r.get('URL')
            })
        return rows[:limit]
    except Exception:
        return []

def get_news_from_newsapi(api_key, query="crypto OR bitcoin OR ethereum", page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        items = []
        for art in data.get("articles", []):
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
