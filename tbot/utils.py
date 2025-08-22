# news/utils.py
import json
import requests
from transformers import pipeline
from .models import NewsItem
from datetime import datetime
from django.utils.timezone import make_aware, is_naive
import math
from django.utils import timezone

def get_news(api_key):
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {'auth_token': api_key}
    all_news = []

    while url:
        response = requests.get(url, params=params)
        data = response.json()
        all_news.extend(data['results'])
        url = data.get('next')
        params = {}

    return all_news

def analyze_sentiment(text):
    sentiment_pipeline = pipeline('sentiment-analysis', model='ProsusAI/finbert', framework='pt')
    analysis = sentiment_pipeline(text)
    return analysis[0]['label'], analysis[0]['score']

def fetch_and_save_news(api_key, latest_timestamp=None):
    news_data = get_news(api_key)
    for post in news_data:
        # Skip news items older than the latest timestamp
        published_at = datetime.strptime(post.get('published_at'), '%Y-%m-%dT%H:%M:%SZ')
        if is_naive(published_at):
            published_at = make_aware(published_at)
        if latest_timestamp and published_at <= latest_timestamp:
            continue

        # Check if news item already exists based on title
        if NewsItem.objects.filter(title=post.get('title')).exists():
            continue

        currencies = [f"{currency.get('code')} - {currency.get('title')}" for currency in post.get('currencies', [])]
        sentiment, score = analyze_sentiment(post.get('title'))

        NewsItem.objects.create(
            title=post.get('title'),
            published_at=published_at,
            currencies=', '.join(currencies),
            sentiment=sentiment,
            score=score
        )

def calculate_market_sentiment(news_items, coin_focus=None):
    now = timezone.now()
    total_weight = 0
    weighted_sum = 0
    
    for item in news_items:
        # 1. Base sentiment direction (-1, 0, or 1)
        if item.sentiment == 'positive':
            direction = 1
        elif item.sentiment == 'negative':
            direction = -1
        else:
            direction = 0
            
        # 2. Calculate recency weight
        hours_old = (now - item.published_at).total_seconds() / 3600
        recency_weight = 1.0 if hours_old < 3 else math.exp(-0.03 * hours_old)
        
        # 3. Calculate coin relevance weight
        if coin_focus is None:
            # Overall market sentiment
            if 'BTC' in item.currencies:
                coin_weight = 1.0  # BTC news gets full weight
            else:
                coin_weight = 0.7  # Other coins get 0.7 weight
                
        else:
            # Specific coin sentiment - only include news about this coin
            if coin_focus in item.currencies:
                if 'BTC' in item.currencies:
                    coin_weight = 1.0  # News about both the target coin and BTC
                else:
                    coin_weight = 0.7  # News only about the target coin
            else:
                coin_weight = 0  # Skip news not about the target coin
        
        # 4. Calculate final weight and add to total
        final_weight = recency_weight * coin_weight
        
        if final_weight > 0:  # Only count if the weight is positive
            weighted_sum += direction * final_weight
            total_weight += final_weight
    
    # Calculate final sentiment score (-1 to +1)
    if total_weight > 0:
        final_score = weighted_sum / total_weight
        
        if final_score > 0.5:
            category = "bullish"
        elif final_score < -0.5:
            category = "bearish"
        else:
            category = "neutral"
            
        return {
            "score": final_score,
            "category": category,
            "news_count": len(news_items)
        }
    else:
        return {
            "score": 0,
            "category": "neutral",
            "news_count": len(news_items)
        }