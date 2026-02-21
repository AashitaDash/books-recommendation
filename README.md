# 📚 Hybrid Book Recommendation System

An AI-powered intelligent book recommendation system built using metadata-based hybrid ranking, clustering, and content similarity techniques.

---

## 🚀 Project Overview

This project implements a **Hybrid Book Recommendation System** that ranks and recommends books based on:

- 📊 Average Rating
- ⭐ Popularity (ratings count)
- 💬 Engagement (text reviews count)
- 📖 Content similarity (TF-IDF on title & author)
- 🧠 Clustering using KMeans

The system combines rule-based ranking with unsupervised machine learning to generate high-quality recommendations.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Cleaned missing values
- Removed column inconsistencies
- Converted numeric columns
- Normalized features using MinMaxScaler

### 2️⃣ Feature Engineering
Created normalized features:
- `norm_rating`
- `norm_popularity`
- `norm_reviews`

### 3️⃣ Hybrid Ranking Formula

Final Hybrid Score:


Hybrid Score =
0.5 × normalized_rating +
0.3 × normalized_ratings_count +
0.2 × normalized_text_reviews_count


This balances:
- Book quality
- Popularity
- Reader engagement

---

### 4️⃣ Content-Based Similarity

- TF-IDF vectorization on:

title + author

- Cosine similarity used to find similar books
- Optimized to compute row-wise similarity (no full NxN matrix)

---

### 5️⃣ Clustering

- KMeans clustering on normalized features
- PCA used for 2D visualization
- Books grouped into behavioral clusters

---

## 📊 Evaluation Metrics

### Precision@K

Measures relevance of top-K recommendations.

Result:

Precision@10 ≈ 0.8


Meaning:
80% of top 10 recommendations are relevant.

---

### Recall@K

Lower due to large relevant set in dataset.
In top-K recommendation systems, precision is more meaningful.

---

## 🖥️ Frontend

Built using:
- HTML
- Tailwind CSS
- JavaScript (Fetch API)
- Modal popup for similar books

Features:
- Dynamic search
- Loading animation
- Similar books popup
- Responsive design

---

## 🛠️ Tech Stack

Backend:
- Flask
- Pandas
- NumPy
- Scikit-learn
- Gunicorn

Frontend:
- HTML
- Tailwind CSS
- JavaScript

Deployment:
- Render (Cloud)

---

## 📁 Project Structure


project/
│
├── app.py
├── books.csv
├── requirements.txt
├── Procfile
├── runtime.txt
│
└── templates/
└── index.html


---

## ▶️ How to Run Locally

1. Clone the repository
2. Install dependencies:


pip install -r requirements.txt


3. Run:


python app.py


4. Open:


http://127.0.0.1:5000


---

## 🌍 Deployment

Deployed using:

- Gunicorn WSGI server
- Render Cloud Platform

Start command:

gunicorn app:app


---

## 🎓 Academic Value

This project demonstrates:

- Data preprocessing
- Feature engineering
- Hybrid ranking systems
- Unsupervised machine learning (KMeans)
- Evaluation metrics (Precision@K)
- Full-stack integration
- Cloud deployment

---

## 📌 Future Improvements

- Add collaborative filtering
- Add user-based recommendations
- Add supervised rating prediction model
- Add deep learning embeddings
- Improve personalization

---

## 👩‍💻 Author

Aashita Dash  
B.Tech Computer Science  

---

## 📜 License

This project is for academic and educational purposes.
