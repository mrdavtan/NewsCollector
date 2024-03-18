import os
import json
import feedparser
import uuid
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
import urllib.parse
import re
import time
from newspaper import Article
import dateutil.parser
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
from unidecode import unidecode

class Scraper:
    def __init__(self, sources, news_date):
        self.sources = sources
        self.news_date = news_date
        self.articles_dir = 'articles'
        self.url_to_uuid = {}

    def check_robots_permission(self, url):
        """Check if the bot is allowed to scrape the given URL based on robots.txt."""
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"

        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except UnicodeDecodeError:
            print(f"Failed to decode robots.txt for {base_url}. Assuming scraping is allowed.")
            return True

        # Assuming the user-agent of your bot is 'MyBot'
        return rp.can_fetch('MyBot', url)

    def generate_uuid_for_article(self, article_url):
        """
        Generate or retrieve a UUID for the article URL.
        """
        if article_url not in self.url_to_uuid:
            # Assign a new UUID for new articles
            self.url_to_uuid[article_url] = uuid.uuid4().hex
        return self.url_to_uuid[article_url]

    def abbreviate_source_name(self, source_name):
        """Abbreviate the source name to no more than 8 letters, using an acronym or truncation."""
        # Split the source name into words and take the first letter of each to form an acronym
        words = source_name.split()
        if len(words) > 1:
            acronym = ''.join(word[0] for word in words).upper()
            # Use the acronym if it's within the limit, otherwise truncate
            return acronym[:8] if len(acronym) <= 8 else acronym[:8]
        else:
            # For a single word, simply truncate to the limit
            return source_name[:8].upper()

    def scrape(self):
        os.makedirs(self.articles_dir, exist_ok=True)
        articles_list = []
        try:
            for source, content in self.sources.items():
                print(f"Processing source: {source}")
                for url in content['rss']:
                    feed = feedparser.parse(url)
                    for entry in feed.entries:
                        if hasattr(entry, 'published'):
                            article_date = dateutil.parser.parse(entry.published)
                            if article_date.strftime('%Y-%m-%d') == str(self.news_date):
                                early_article_details = {
                                    'source': source,
                                    'url': getattr(entry, 'link', 'No URL Available'),
                                    'title': getattr(entry, 'title', 'No Title Available'),
                                    'description': getattr(entry, 'description', 'No Description Available'),
                                    'date': article_date.strftime('%Y-%m-%d'),
                                    'time': '00:00:00'
                                }
                                self.save_article_as_json(early_article_details, self.articles_dir)
                                print(f"Pre-saved article details for {early_article_details['title']}")

                                robots_permission = self.check_robots_permission(entry.link)
                                try:
                                    headers = {'User-Agent': 'Mozilla/5.0'}
                                    response = requests.get(entry.link, headers=headers)
                                    if response.status_code == 200:
                                        article = Article(entry.link)
                                        article.set_html(response.text)
                                        article.parse()
                                        article.nlp()
                                        article_details = {
                                            'source': source,
                                            'url': entry.link,
                                            'date': article_date.strftime('%Y-%m-%d'),
                                            'time': article_date.strftime('%H:%M:%S %Z'),
                                            'title': article.title,
                                            'body': article.text,
                                            'summary': article.summary,
                                            'keywords': article.keywords,
                                            'image_url': article.top_image,
                                            'robots_permission': robots_permission
                                        }
                                        articles_list.append(article_details)
                                        self.save_article_as_json(article_details, self.articles_dir)
                                        print(f"Saved article: {article.title}")
                                    else:
                                        print(f"Request failed with status code: {response.status_code}")
                                except Exception as e:
                                    print(e)
                                    print('continuing...')
                                time.sleep(1)
            return articles_list
        except Exception as e:
            raise Exception(f'Error in "Scraper.scrape()": {e}')

    def save_article_as_json(self, article, directory):
        # Ensure the article has a unique identifier ('id') as its first property
        article_id = self.generate_uuid_for_article(article['url'])
        article_with_id = {'id': article_id}
        article_with_id.update(article)

        # Apply the shortening logic to the source name
        words = article['source'].split()
        if len(words) >= 2:
            source_name_abbreviation = words[0][:3] + "_" + words[1][:2]
        else:
            source_name_abbreviation = words[0][:5]
        source_name_abbreviation = re.sub(r'[\\/*?:"<>|]', '_', source_name_abbreviation)

        # Sanitize the title
        sanitized_title = re.sub(r'[\\/*?:"<>|]', '_', article['title'])
        sanitized_title = re.sub(r'\s+', '_', sanitized_title)[:50]

        # Format the date and time for the filename
        formatted_date = article['date'].replace('-', '') + '_' + article['time'].split(':')[0] + article['time'].split(':')[1]

        # Generate the filename using the abbreviated source name, sanitized title, and formatted date/time
        filename = f"{source_name_abbreviation}_{sanitized_title}_{formatted_date}.json"
        filepath = os.path.join(directory, filename)

        # Write the article details, including 'id', to a JSON file
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(article_with_id, file, ensure_ascii=False, indent=4)

        print(f"Article saved: {filepath}")

def load_sources(file):
    try:
        with open(file) as data:
            sources = json.load(data)
        print(f'INFO: Using custom "{file}" as source file.')
        return sources
    except:
        raise Exception(f'Error in "Helper.load_sources()"')

def write_dataframe(sources):
    try:
        df = pd.json_normalize(sources)
        return df
    except:
        raise Exception(f'Error in "Helper.write_dataframe()"')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def clean_dataframe(df):
    try:
        df = df[df.title != '']
        df = df[df.body != '']
        df = df[df.image_url != '']
        df = df[df.title.str.count('\s+').ge(3)]
        df = df[df.body.str.count('\s+').ge(20)]
        return df
    except:
        raise Exception(f'Error in "Helper.clean_dataframe()"')

def clean_articles(df):
    try:
        # Drop Duplicates
        df = (df.drop_duplicates(subset=["title", "source"])).sort_index()
        df = (df.drop_duplicates(subset=["body"])).sort_index()
        df = (df.drop_duplicates(subset=["url"])).sort_index()
        df = df.reset_index(drop=True)

        # Make all letters lower case
        df['clean_body'] = df['body'].str.lower()

        # Filter out the stopwords, punctuation, and digits
        df['clean_body'] = [remove_stopwords(x).translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits)) for x in df['clean_body']]

        # Remove sources
        sources_set = [x.lower() for x in set(df['source'])]
        sources_to_replace = dict.fromkeys(sources_set, "")
        df['clean_body'] = (df['clean_body'].replace(sources_to_replace, regex=True))

        # Unidecode all characters
        df['clean_body'] = df['clean_body'].apply(unidecode)

        # Tokenize
        df['clean_body'] = df['clean_body'].apply(word_tokenize)

        # Stem words
        stemmer = SnowballStemmer(language='english')
        df['clean_body'] = df["clean_body"].apply(lambda x: [stemmer.stem(y) for y in x])
        df['clean_body'] = df["clean_body"].apply(lambda x: ' '.join([word for word in x]))

        return df
    except:
        raise Exception(f'Error in "Helper.clean_articles()"')

# Load the RSS sources from the JSON file
rss_feeds = load_sources('tech_sources.json')

# Example usage
news_date = datetime.today().strftime('%Y-%m-%d')
scraper = Scraper(rss_feeds, news_date)
articles = scraper.scrape()

# Create a DataFrame from the scraped articles
df = write_dataframe(articles)

# Clean and preprocess the DataFrame
df = clean_dataframe(df)
df = clean_articles(df)
