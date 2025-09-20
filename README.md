# 🎬 Hybrid Movie Recommendation System

A **hybrid movie recommendation system** that combines **content-based filtering (TF-IDF)** and **machine learning (XGBoost)** with **real-time TMDB API integration** to recommend movies based on any user query. This project is designed as a **portfolio-ready interactive app** with Streamlit and a visual Looker Studio dashboard.

---

## **Features**

- Recommend movies based on **any input movie title**, even if it’s not in the original MovieLens dataset.  
- Combines **TF-IDF vector similarity** with **XGBoost predicted ratings** for hybrid recommendations.  
- Displays **movie posters, genres, and predicted ratings** in a clean Streamlit interface.  
- Fully deployable via **Streamlit Cloud** with **TMDB API key stored securely**.

---

## **Tech Stack**

- **Python Libraries:** `pandas`, `scikit-learn`, `xgboost`, `requests`, `streamlit`  
- **Datasets:** MovieLens 100K/25M dataset  
- **APIs:** TMDB API for fetching movie details and posters  
- **Deployment:** Streamlit Cloud  

---

## **How It Works**

1. **Data Preprocessing:**  
   - Load MovieLens dataset (`movies.csv`).  
   - Combine `title` + `genres` for TF-IDF vectorization (max 5000 features).  

2. **Hybrid Model:**  
   - **Content-Based Filtering:** Compute **cosine similarity** between movies using TF-IDF.  
   - **XGBoost Regression:** Predict ratings for candidate movies to rank recommendations.  

3. **TMDB API Integration:**  
   - Accepts **any movie query**.  
   - Fetches **overview, genres, and poster URL** from TMDB.  
   - Converts TMDB genres to names using the genre mapping API.  
   - Computes similarity with MovieLens dataset for recommendations.  

4. **Streamlit UI:**  
   - Input movie title → Returns **top-N recommended movies** with:  
     - Poster  
     - Title  
     - Genres  
     - Predicted rating  
---

## **Usage**

### **Local Setup**

```bash
# Clone repo
git clone https://github.com/zia9571/Hybrid-Movie-Recommendation-System.git
cd Hybrid-Movie-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
