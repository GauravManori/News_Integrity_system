#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Load the Sentence-BERT model
model = SentenceTransformer('all-mpnet-base-v2')

# RSS Feed URLs for various news sources
news_sources = {
    "Times of India": 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
    "NDTV": 'https://feeds.feedburner.com/ndtvnews-top-stories',
    "The Hindu": 'https://www.thehindu.com/feeder/default.rss'
}

# Indian Express Scraper URL
indian_express_url = 'https://indianexpress.com/section/india/'


# Function to fetch articles from RSS feeds
def fetch_news(feed_url, source_name):
    feed = feedparser.parse(feed_url)
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    articles = []

    for entry in feed.entries:
        if hasattr(entry, 'published_parsed'):
            published_date = pd.Timestamp(*entry.published_parsed[:3]).strftime('%Y-%m-%d')
            if published_date == today:
                articles.append({
                    "title": entry.title,
                    "summary": entry.summary,
                    "link": entry.link,
                    "published": published_date,
                    "source": source_name
                })
    return articles


# Function to fetch articles from Indian Express
def fetch_indian_express_news(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    news_articles = []
    today = datetime.now().strftime('%Y-%m-%d')

    for article in soup.find_all('div', class_='articles'):
        title_tag = article.find('h2')
        link_tag = article.find('a')

        if title_tag and link_tag:
            title = title_tag.text.strip()
            link = link_tag['href']
            summary_tag = article.find('p')
            summary = summary_tag.text.strip() if summary_tag else "No summary available"

            news_articles.append({
                'title': title,
                'summary': summary,
                'link': link,
                'published': today,
                'source': 'Indian Express'
            })
    return news_articles


# Function to deduplicate articles using Sentence-BERT embeddings
def deduplicate_articles(articles):
    articles_df = pd.DataFrame(articles)
    articles_df['embedding'] = articles_df['summary'].apply(
        lambda x: model.encode(x, normalize_embeddings=True) if isinstance(x, str) and x.strip() else None
    )

    unique_articles = []
    duplicates = []

    for i, row in articles_df.iterrows():
        if row['embedding'] is None:
            continue
        is_duplicate = False
        for unique_article in unique_articles:
            similarity = util.pytorch_cos_sim(row['embedding'], unique_article['embedding']).item()
            if similarity >= 0.8:  # Similarity threshold
                duplicates.append({
                    "original_title": unique_article['title'],
                    "duplicate_title": row['title'],
                    "similarity_score": similarity,
                    "source": row['source']
                })
                is_duplicate = True
                break
        if not is_duplicate:
            unique_articles.append(row)

    unique_articles_df = pd.DataFrame(unique_articles)
    duplicates_df = pd.DataFrame(duplicates)
    return unique_articles_df, duplicates_df


# Streamlit App
st.title("News Deduplication App")

# Dropdown for selecting news sources
selected_sources = st.multiselect(
    "Select News Sources",
    options=list(news_sources.keys()) + ["Indian Express"],
    default=list(news_sources.keys())
)

if st.button("Run"):
    all_articles = []
    source_stats = {}

    # Fetch articles from RSS sources
    for source, url in news_sources.items():
        if source in selected_sources:
            articles = fetch_news(url, source)
            source_stats[source] = len(articles)
            all_articles.extend(articles)

    # Fetch articles from Indian Express if selected
    if "Indian Express" in selected_sources:
        indian_express_articles = fetch_indian_express_news(indian_express_url)
        source_stats["Indian Express"] = len(indian_express_articles)
        all_articles.extend(indian_express_articles)

    # Check if no articles were fetched
    if len(all_articles) == 0:
        st.write("No articles fetched. Please check your sources or try again later.")
    else:
        # Deduplicate articles
        unique_articles_df, duplicates_df = deduplicate_articles(all_articles)

        # Display statistics
        st.header("Statistics")
        st.write(f"Total Articles Fetched: {len(all_articles)}")
        for source, count in source_stats.items():
            st.write(f"{source}: {count} articles")
        st.write(f"Number of Duplicates: {len(duplicates_df)}")

        # Show duplicates
        if not duplicates_df.empty:
            st.subheader("Duplicate Articles")
            st.write(duplicates_df[["original_title", "duplicate_title", "similarity_score", "source"]])
        else:
            st.write("No duplicates found.")

        # Show deduplicated articles
        st.subheader("Deduplicated Articles")
        st.write(unique_articles_df[["title", "summary", "published", "source"]])

        # Download options
        st.download_button(
            label="Download Deduplicated Articles",
            data=unique_articles_df.to_csv(index=False),
            file_name="deduplicated_articles.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Duplicate Articles Log",
            data=duplicates_df.to_csv(index=False),
            file_name="duplicate_articles.csv",
            mime="text/csv"
        )

