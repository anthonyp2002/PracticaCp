import os
import praw
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from googletrans import Translator

# Cargar las variables de entorno
load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)

print(f"Autenticado como: {reddit.user.me()}")

translator = Translator(timeout=20)
def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^\w\s,¡!¿?]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'[^\w\s]', '', text)

    return text

search_terms = ['What’s your opinion on Temu Online Shopping?']
comments_data = []

for submission in reddit.subreddit('all').search(' '.join(search_terms), limit=10):
    print(f"Buscando en: {submission.title}")
    submission.comments.replace_more(limit=0)

    for comment in submission.comments.list():

        soup = BeautifulSoup(comment.body, 'html.parser')
        clean_comment = soup.get_text()

        cleaned_comment = clean_text(clean_comment)

        if cleaned_comment and 'en' in translator.detect(cleaned_comment).lang:
            try:
                cleaned_comment = translator.translate(cleaned_comment, src='en', dest='es').text
            except Exception as e:
                print(f"Error al traducir el comentario: {e}")
                continue  

        comments_data.append({
            'Comment': cleaned_comment
        })

df = pd.DataFrame(comments_data)
df.to_csv('controversial_comments.csv', index=False)

print(df.head())
