import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')
LOC_DIR=os.path.dirname(os.path.abspath(__file__))
movies = pd.read_csv(os.path.join(LOC_DIR,"movies.csv"))
ratings = pd.read_csv(os.path.join(LOC_DIR,"ratings.csv"))
tags = pd.read_csv(os.path.join(LOC_DIR,"tags.csv"))

# avg ratings
avg_ratings = (ratings.groupby("movieId")["rating"]
               .agg(avg_rating="mean", rating_count="count")
               .reset_index())

# clean tags
tags = tags.dropna(subset=["tag"]).copy()
tags["tag"] = tags["tag"].str.lower().str.strip()
tags_grouped = (tags.groupby("movieId")["tag"]
                .apply(lambda x: " ".join(x))
                .reset_index()
                .rename(columns={"tag": "all_tags"}))

#dataframe
df = movies.merge(avg_ratings, on="movieId", how="left")
df = df.merge(tags_grouped, on="movieId", how="left")
df["avg_rating"] = df["avg_rating"].fillna(0)
df["avg_rating"]= df["avg_rating"].round(3)
df["rating_count"] = df["rating_count"].fillna(0)
df["all_tags"] = df["all_tags"].fillna("")
df["genre_clean"] = df["genres"].str.replace("|", " ", regex=False).str.lower()


# vader sentiment
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text.strip():
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_score"] = df["all_tags"].apply(get_sentiment)
df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

pos = (df["sentiment_score"] >= 0.05).sum()
neg = (df["sentiment_score"] <= -0.05).sum()
neu = ((df["sentiment_score"] > -0.05) & (df["sentiment_score"] < 0.05)).sum()

# soup
df["soup"] = (df["genre_clean"] + " " + df["all_tags"]).str.strip()

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["soup"])

# cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df["title"].str.lower().str.strip())

# recommendation function
def recommend(movie_title, top_n=10):
    title_key = movie_title.lower().strip()

    if title_key not in indices:
        matches = [t for t in indices.index if title_key in t]
        if not matches:
            print("Movie not found! Check spelling and try again,or No recommendations found in the current database please try later ;)...")
            return None
        title_key = matches[0]

    idx = indices[title_key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]

    cand_idx = [i[0] for i in sim_scores[:top_n*3]]
    candidates = df.iloc[cand_idx].copy()
    candidates["cosine_score"] = [i[1] for i in sim_scores[:top_n*3]]
    candidates["sentiment_boost"] = candidates["sentiment_score"].apply(lambda s: s if s > 0 else 0)

    max_r = df["avg_rating"].max()
    candidates["norm_rating"] = candidates["avg_rating"] / max_r
    candidates["Blaten_score"] = (
        candidates["cosine_score"] +
        0.15 * candidates["sentiment_boost"] +
        0.05 * candidates["norm_rating"]
    )

    return (candidates
            .sort_values("Blaten_score", ascending=False)
            .head(top_n)[["title", "genres", "avg_rating", "sentiment_label", "Blaten_score"]]
            .reset_index(drop=True))

