# ===============================
# IMPORT LIBRARIES
# ===============================
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ===============================
# LOAD & CLEAN DATA
# ===============================
df = pd.read_csv("books.csv", engine="python", on_bad_lines="skip")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

# Select required columns
df = df[['bookID','title','authors','average_rating',
         'ratings_count','text_reviews_count',
         'num_pages','language_code','publication_date']]

df = df.dropna()

# Convert numeric columns
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')

df = df.dropna()
df = df.reset_index(drop=True)

# ===============================
# NORMALIZE FEATURES
# ===============================
scaler = MinMaxScaler()

df[['norm_rating',
    'norm_popularity',
    'norm_reviews']] = scaler.fit_transform(
    df[['average_rating','ratings_count','text_reviews_count']]
)

# ===============================
# HYBRID SCORE
# ===============================
df['hybrid_score'] = (
    0.5 * df['norm_rating'] +
    0.3 * df['norm_popularity'] +
    0.2 * df['norm_reviews']
)

# ===============================
# TF-IDF FOR CONTENT SIMILARITY
# ===============================
df['combined_text'] = df['title'] + " " + df['authors']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# ===============================
# SEARCH FUNCTION (HYBRID RANKING)
# ===============================
def search_books(query, top_k=10):

    query_lower = query.lower()

    # Filter books containing query in title or author
    filtered_df = df[
        df['title'].str.lower().str.contains(query_lower, na=False) |
        df['authors'].str.lower().str.contains(query_lower, na=False)
    ]

    if filtered_df.empty:
        # If no match, return top hybrid books
        top_books = df.sort_values(by='hybrid_score', ascending=False).head(top_k)
    else:
        top_books = filtered_df.sort_values(by='hybrid_score', ascending=False).head(top_k)

    results = []

    for _, row in top_books.iterrows():
        results.append({
            "title": row['title'],
            "author": row['authors'],
            "rating": round(float(row['average_rating']), 2),
            "score": round(float(row['hybrid_score']), 4)
        })

    return results


# ===============================
# SIMILARITY FUNCTION
# ===============================
def recommend_similar(book_title, top_n=5):

    idx = df[df['title'] == book_title].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    book_vector = tfidf_matrix[idx]

    similarities = cosine_similarity(book_vector, tfidf_matrix).flatten()

    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    results = []

    for i in similar_indices:
        results.append({
            "title": df.iloc[i]['title'],
            "author": df.iloc[i]['authors'],
            "rating": round(float(df.iloc[i]['average_rating']), 2)
        })

    return results


# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.json
    query = data.get("query")
    results = search_books(query)
    return jsonify(results)


@app.route("/similar", methods=["POST"])
def similar_api():
    data = request.json
    book_title = data.get("title")
    results = recommend_similar(book_title)
    return jsonify(results)


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)