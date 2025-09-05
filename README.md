# 🎬 MovieLens-25M Recommender System

This project demonstrates building a **hybrid recommender system** using the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).  
The pipeline combines **Matrix Factorization (SVD/ALS)** for candidate generation and **Gradient Boosted Trees (LightGBM/XGBoost)** for ranking, with rich item and user features.

---

## 🚀 Objectives
- Showcase practical **recommender system design** from baseline to hybrid models.
- Learn both **collaborative filtering** (latent factors) and **content-based features** (genres, popularity, recency).
- Evaluate improvements using **Precision@K, Recall@K, and NDCG**.

---

## 📂 Project Structure

movielens-25m-recsys/
├─ notebooks/
│  ├─ 01_eda.ipynb            # Exploratory Data Analysis (EDA)
│  ├─ 02_mf_baseline.ipynb    # Matrix Factorization (SVD/ALS)
│  ├─ 03_hybrid_ranker.ipynb  # Hybrid model with LightGBM/XGBoost
├─ src/
│  ├─ data_utils.py           # k-core filtering, train/valid/test split
│  ├─ mf.py                   # Matrix Factorization utilities
│  ├─ ranker.py               # Hybrid model training & inference
│  ├─ metrics.py              # Precision@K, Recall@K, NDCG
├─ outputs/
│  ├─ data/                   # Parquet files, label tables, features
│  └─ figure/                 # Result figures (Precision/Recall/NDCG plots)
└─ README.md


---

## 📊 Pipeline Overview
1. **Data Preparation**
   - Filter with **k-core** (remove inactive users/items).
   - Time-based **train/valid/test split**.
   - Extract item features (popularity, recency, genres).
   - Build user features (average genre preference).

2. **Baseline Model (MF)**
   - Apply **SVD/ALS** to learn user/item embeddings.
   - Generate Top-K candidate items per user.

3. **Hybrid Ranking**
   - Features:
     - MF score
     - Genre similarity
     - Item popularity
     - Recency
   - Train **LightGBM/XGBoost** to re-rank candidates.

4. **Evaluation**
   - Compute **Precision@K, Recall@K, NDCG**.
   - Compare **MF baseline vs Hybrid model**.

---

## 🖼️ Example Outputs
- Precision vs K  
- Recall vs K  
- NDCG vs K  
- Feature importance (e.g., MF score vs. genres vs. recency)

---

## 💡 Insights
- **Hybrid > MF baseline** across all metrics.
- **Genre similarity** improves personalization.
- **Recency** ensures fresh recommendations.
- **Popularity** balances niche vs blockbuster movies.

---

## 📌 Tech Stack
- Python (pandas, numpy, scipy)
- scikit-learn
- LightGBM / XGBoost
- Matplotlib / Seaborn

