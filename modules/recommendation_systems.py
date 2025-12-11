import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("🎬 Recommendation Systems")
    
    st.markdown("""
    How Netflix knows what you want to watch, Amazon knows what you want to buy!
    """)
    
    tabs = st.tabs([
        "📚 Introduction",
        "👥 Collaborative Filtering",
        "📝 Content-Based",
        "🔢 Matrix Factorization",
        "🎮 Build Your Own"
    ])
    
    # TAB 1: Introduction
    with tabs[0]:
        st.header("What are Recommendation Systems?")
        
        st.markdown("""
        **Recommendation Systems** predict user preferences and suggest relevant items.
        
        **Examples:**
        - 🎬 Netflix: Movie recommendations
        - 🛒 Amazon: Product recommendations
        - 🎵 Spotify: Music discovery
        - 📱 TikTok: Video feed
        - 📰 News: Article suggestions
        """)
        
        st.subheader("Types of Recommendation Systems")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👥 Collaborative Filtering")
            st.info("""
            "Users who liked X also liked Y"
            
            Uses: User-item interactions
            
            **Pros:** No item features needed
            **Cons:** Cold start problem
            """)
            
        with col2:
            st.markdown("### 📝 Content-Based")
            st.info("""
            "Because you liked X, try similar Y"
            
            Uses: Item features
            
            **Pros:** No cold start for items
            **Cons:** Limited diversity
            """)
            
        with col3:
            st.markdown("### 🔀 Hybrid")
            st.info("""
            Combine both approaches
            
            Uses: Interactions + Features
            
            **Pros:** Best of both worlds
            **Cons:** More complex
            """)
        
        st.subheader("The Cold Start Problem")
        
        st.warning("""
        **Cold Start:** How to recommend for new users or new items with no history?
        
        **Solutions:**
        - Ask for preferences during onboarding
        - Use content-based for new items
        - Use demographic info for new users
        - Popular items as fallback
        """)
    
    # TAB 2: Collaborative Filtering
    with tabs[1]:
        st.header("👥 Collaborative Filtering")
        
        st.markdown("""
        Based on the idea: **"Similar users like similar items"**
        """)
        
        st.subheader("User-Item Matrix")
        
        # Create sample matrix
        users = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        movies = ["Inception", "Titanic", "Matrix", "Notebook", "Avengers"]
        
        np.random.seed(42)
        ratings = np.array([
            [5, 3, 5, 1, 5],
            [4, 2, 4, 1, 4],
            [2, 5, 1, 5, 2],
            [1, 4, 2, 5, 1],
            [5, 0, 4, 0, 5],  # 0 = not rated
        ])
        
        df_matrix = pd.DataFrame(ratings, index=users, columns=movies)
        
        st.dataframe(df_matrix.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=5))
        st.caption("0 = Not rated yet (we want to predict these)")
        
        st.subheader("Two Approaches")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### User-Based CF")
            st.code("""
# Find similar users
similarity = cosine_similarity(user_ratings)

# Predict: weighted average of similar users' ratings
def predict(user, item):
    similar_users = get_top_k_similar(user)
    weighted_sum = sum(sim * rating for sim, rating in similar_users)
    return weighted_sum / sum(similarities)
            """, language="python")
            
        with col2:
            st.markdown("### Item-Based CF")
            st.code("""
# Find similar items
similarity = cosine_similarity(item_ratings.T)

# Predict: weighted average of similar items' ratings
def predict(user, item):
    similar_items = get_top_k_similar(item)
    weighted_sum = sum(sim * user_rating[item] for sim, item in similar_items)
    return weighted_sum / sum(similarities)
            """, language="python")
        
        st.subheader("Similarity Metrics")
        
        st.markdown("""
        | Metric | Formula | Best For |
        |--------|---------|----------|
        | **Cosine Similarity** | cos(θ) = A·B / (‖A‖‖B‖) | Sparse data |
        | **Pearson Correlation** | Centered cosine | Rating scales |
        | **Jaccard Similarity** | Intersection / Union | Binary (like/dislike) |
        | **Euclidean Distance** | √Σ(a-b)² | Dense data |
        """)
        
        st.latex(r"Cosine(A, B) = \frac{A \cdot B}{\|A\| \|B\|}")
    
    # TAB 3: Content-Based
    with tabs[2]:
        st.header("📝 Content-Based Filtering")
        
        st.markdown("""
        Recommend items similar to what the user already liked, based on **item features**.
        """)
        
        st.subheader("Example: Movie Recommendations")
        
        movies_data = pd.DataFrame({
            'Title': ['Inception', 'Matrix', 'Titanic', 'Notebook', 'Avengers', 'Dark Knight'],
            'Genre': ['Sci-Fi, Thriller', 'Sci-Fi, Action', 'Romance, Drama', 'Romance, Drama', 'Action, Sci-Fi', 'Action, Thriller'],
            'Director': ['Nolan', 'Wachowski', 'Cameron', 'Cassavetes', 'Russo', 'Nolan'],
            'Rating': [8.8, 8.7, 7.9, 7.8, 8.0, 9.0]
        })
        
        st.dataframe(movies_data)
        
        st.subheader("TF-IDF for Text Features")
        
        st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine features into text
movies['features'] = movies['Genre'] + ' ' + movies['Director']

# Create TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['features'])

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)

# Get recommendations
def get_recommendations(movie_title, top_n=5):
    idx = movies[movies['Title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [movies['Title'][i] for i, _ in sim_scores[1:top_n+1]]
        """, language="python")
        
        st.subheader("User Profile")
        
        st.markdown("""
        Build a **user profile** from their liked items:
        
        1. Get feature vectors of all liked items
        2. Average (or weighted average) them
        3. Compare new items to this profile
        """)
        
        st.code("""
def build_user_profile(user_liked_items):
    item_vectors = [get_item_vector(item) for item in user_liked_items]
    user_profile = np.mean(item_vectors, axis=0)
    return user_profile

def recommend(user_profile, all_items, top_n=10):
    scores = [(item, cosine_similarity(user_profile, get_item_vector(item)))
              for item in all_items]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        """, language="python")
    
    # TAB 4: Matrix Factorization
    with tabs[3]:
        st.header("🔢 Matrix Factorization")
        
        st.markdown("""
        Decompose the user-item matrix into **latent factors**.
        """)
        
        st.subheader("The Idea")
        
        st.latex(r"R \approx U \times V^T")
        
        st.markdown("""
        Where:
        - **R**: User-Item rating matrix (m users × n items)
        - **U**: User latent factors (m × k)
        - **V**: Item latent factors (n × k)
        - **k**: Number of latent factors (typically 10-100)
        """)
        
        # Visual
        st.subheader("Visual Decomposition")
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
        
        with col1:
            st.markdown("**R (Ratings)**")
            st.markdown("1000 × 5000")
            st.markdown("Users × Movies")
        
        with col2:
            st.markdown("≈")
        
        with col3:
            st.markdown("**U (User factors)**")
            st.markdown("1000 × 50")
            st.markdown("Users × Latent")
        
        with col4:
            st.markdown("×")
        
        with col5:
            st.markdown("**V^T (Item factors)**")
            st.markdown("50 × 5000")
            st.markdown("Latent × Movies")
        
        st.subheader("SVD (Singular Value Decomposition)")
        
        st.code("""
from scipy.sparse.linalg import svds
import numpy as np

# R is the user-item matrix (fill NaN with 0 or mean)
R = user_item_matrix.fillna(0).values

# Mean center the data
user_means = np.mean(R, axis=1)
R_centered = R - user_means.reshape(-1, 1)

# SVD
U, sigma, Vt = svds(R_centered, k=50)

# Reconstruct
sigma_diag = np.diag(sigma)
predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)
        """, language="python")
        
        st.subheader("What are Latent Factors?")
        
        st.markdown("""
        Latent factors are hidden features that explain preferences:
        
        **Example latent factors for movies:**
        - Factor 1: Action vs Romance
        - Factor 2: Modern vs Classic
        - Factor 3: Complex plot vs Simple
        - Factor 4: Male-oriented vs Female-oriented
        
        **The model learns these automatically!**
        """)
        
        st.subheader("Modern Approaches")
        
        st.markdown("""
        | Method | Description |
        |--------|-------------|
        | **ALS (Alternating Least Squares)** | Scalable, works with implicit data |
        | **Neural Collaborative Filtering** | Deep learning for CF |
        | **Variational Autoencoders** | Probabilistic approach |
        | **Graph Neural Networks** | Model user-item as graph |
        | **Two-Tower Models** | Separate user/item encoders |
        """)
    
    # TAB 5: Build Your Own
    with tabs[4]:
        st.header("🎮 Build Your Own Recommender")
        
        st.subheader("Step 1: Rate Some Movies")
        
        movies = ["Inception", "Titanic", "The Matrix", "The Notebook", "Avengers", "The Dark Knight", "Forrest Gump", "Interstellar"]
        genres = ["Sci-Fi", "Romance", "Sci-Fi", "Romance", "Action", "Action", "Drama", "Sci-Fi"]
        
        user_ratings = {}
        
        cols = st.columns(4)
        for i, (movie, genre) in enumerate(zip(movies, genres)):
            with cols[i % 4]:
                rating = st.slider(f"🎬 {movie}", 0, 5, 0, key=f"movie_{i}")
                st.caption(genre)
                if rating > 0:
                    user_ratings[movie] = (rating, genre)
        
        if st.button("🎯 Get Recommendations"):
            if len(user_ratings) < 2:
                st.warning("Please rate at least 2 movies!")
            else:
                st.subheader("Step 2: Your Taste Profile")
                
                # Calculate genre preferences
                genre_scores = {}
                for movie, (rating, genre) in user_ratings.items():
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating)
                
                genre_avg = {g: np.mean(scores) for g, scores in genre_scores.items()}
                
                fig = go.Figure(go.Bar(
                    x=list(genre_avg.keys()),
                    y=list(genre_avg.values()),
                    marker_color='steelblue'
                ))
                fig.update_layout(title="Your Genre Preferences", yaxis_title="Average Rating")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Step 3: Recommendations")
                
                # Simple content-based recommendation
                favorite_genre = max(genre_avg, key=genre_avg.get)
                
                # Extended movie list for recommendations
                all_movies = {
                    "Sci-Fi": ["Blade Runner", "Ex Machina", "Arrival", "The Martian"],
                    "Romance": ["La La Land", "The Fault in Our Stars", "Pride & Prejudice"],
                    "Action": ["John Wick", "Mad Max: Fury Road", "Mission Impossible"],
                    "Drama": ["The Shawshank Redemption", "The Godfather", "Schindler's List"]
                }
                
                st.success(f"Based on your love for **{favorite_genre}**, you might enjoy:")
                
                recommendations = all_movies.get(favorite_genre, ["No recommendations available"])
                
                for i, movie in enumerate(recommendations, 1):
                    st.markdown(f"**{i}. {movie}** 🎬")
                
                st.info("""
                💡 **How this works:**
                1. Calculated your average rating per genre
                2. Found your favorite genre
                3. Recommended popular movies from that genre
                
                Real systems use much more sophisticated approaches with millions of users and items!
                """)
