import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import html

import re
import random

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_ai_news(limit=10):
    """
    Fetches real-time AI news from Google News RSS feed.
    """
    url = "https://news.google.com/rss/search?q=Artificial+Intelligence+when:2d&hl=en-US&gl=US&ceid=US:en"
    
    # Fallback images for AI news
    fallback_images = [
        "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=800", # AI Brain
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&q=80&w=800", # AI Chip
        "https://images.unsplash.com/photo-1676299081847-824916de030a?auto=format&fit=crop&q=80&w=800", # Neural Net
        "https://images.unsplash.com/photo-1535378437327-b71494669e96?auto=format&fit=crop&q=80&w=800", # Robot Hand
        "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?auto=format&fit=crop&q=80&w=800", # Cyber Girl
        "https://images.unsplash.com/photo-1617791160505-6f00504e3519?auto=format&fit=crop&q=80&w=800", # Data Vis
        "https://images.unsplash.com/photo-1555255707-c07966088b7b?auto=format&fit=crop&q=80&w=800", # Code
        "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800"  # Circuit
    ]

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        items = root.findall('.//item')
        
        news_items = []
        seen_titles = set()
        
        for item in items:
            if len(news_items) >= limit:
                break
                
            title = item.find('title').text
            # Cleanup title (remove source usually at the end)
            clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title
            
            # Deduplication
            is_duplicate = False
            for seen in seen_titles:
                if clean_title.lower() in seen.lower() or seen.lower() in clean_title.lower():
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
                
            seen_titles.add(clean_title)

            link = item.find('link').text
            pub_date = item.find('pubDate').text
            
            # Extract Image from description if available
            description = item.find('description').text if item.find('description') is not None else ""
            img_match = re.search(r'src="([^"]+)"', description)
            
            image_url = img_match.group(1) if img_match else random.choice(fallback_images)
            
            # Simple date parsing
            try:
                # Format: Mon, 25 Dec 2023 12:00:00 GMT
                dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                date_str = dt.strftime('%b %d, %H:%M')
            except:
                date_str = pub_date[:16] # Fallback
                
            source_elem = item.find('source')
            source = source_elem.text if source_elem is not None else "News"

            news_items.append({
                "title": clean_title,
                "link": link,
                "date": date_str,
                "source": source,
                "image": image_url
            })
            
        return news_items
        
    except Exception as e:
        # Fallback in case of network error
        return [
            {
                "title": "Unable to fetch live news at the moment.",
                "link": "#",
                "date": datetime.now().strftime('%b %d'),
                "source": "System"
            }
        ]
