import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

def show():
    st.title("🛠️ Hands-On Projects")
    
    project = st.selectbox("Select Project:", [
        "1. House Price Predictor (Regression)",
        "2. Iris Flower Classifier (Classification)",
        "3. Sentiment Analyzer (NLP)",
        "4. Handwritten Digit Classifier (Vision)",
        "5. Movie Recommender (Collaborative Filtering)"
    ])
    
    if project == "1. House Price Predictor (Regression)":
        house_price_project()
    elif project == "2. Iris Flower Classifier (Classification)":
        iris_project()
    elif project == "3. Sentiment Analyzer (NLP)":
        sentiment_project()
    elif project == "4. Handwritten Digit Classifier (Vision)":
        digit_classifier_project()
    elif project == "5. Movie Recommender (Collaborative Filtering)":
        recommender_project()

def house_price_project():
    st.header("🏠 House Price Predictor")
    st.markdown("""
    **Goal:** Predict house prices based on features like Size and Number of Bedrooms.
    **Data:** Synthetic real-estate data.
    """)
    
    # 1. Generate Data
    np.random.seed(42)
    n_samples = 500
    size = np.random.normal(1500, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    
    # Price = 100 * Size + 20000 * Bedrooms + Noise
    price = 100 * size + 20000 * bedrooms + np.random.normal(0, 10000, n_samples)
    
    df = pd.DataFrame({'Size (sqft)': size, 'Bedrooms': bedrooms, 'Price ($)': price})
    
    st.subheader("1. Explore Data")
    st.dataframe(df.head())
    
    fig = px.scatter(df, x='Size (sqft)', y='Price ($)', color='Bedrooms', title="Price vs Size")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("2. Train Model")
    if st.button("Train Regressor", key="train_house"):
        X = df[['Size (sqft)', 'Bedrooms']]
        y = df['Price ($)']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        st.success(f"Model Trained! R² Score: {score:.2f}")
        
        st.subheader("3. Make Prediction")
        col1, col2 = st.columns(2)
        u_size = col1.number_input("Size (sqft)", 500, 5000, 1500)
        u_bed = col2.number_input("Bedrooms", 1, 10, 3)
        
        pred = model.predict([[u_size, u_bed]])[0]
        st.metric("Estimated Price", f"${pred:,.2f}")

def iris_project():
    st.header("🌸 Iris Flower Classifier")
    st.markdown("""
    **Goal:** Classify flower species based on petal/sepal dimensions.
    **Data:** The classic Iris dataset.
    """)
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].apply(lambda x: iris.target_names[x])
    
    st.subheader("1. The Data")
    st.write(df.sample(5))
    
    st.subheader("2. Interactive Testing")
    st.write("Adjust the sliders to 'measure' a flower and ask the AI what species it is.")
    
    col1, col2 = st.columns(2)
    sl = col1.slider("Sepal Length", 4.0, 8.0, 5.0)
    sw = col2.slider("Sepal Width", 2.0, 5.0, 3.0)
    pl = col1.slider("Petal Length", 1.0, 7.0, 4.0)
    pw = col2.slider("Petal Width", 0.1, 2.5, 1.0)
    
    # Train Model on the fly
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    
    prediction = model.predict([[sl, sw, pl, pw]])
    proba = model.predict_proba([[sl, sw, pl, pw]])
    
    class_name = iris.target_names[prediction[0]]
    
    st.success(f"Prediction: **{class_name.upper()}**")
    
    # Visual Probability
    prob_df = pd.DataFrame(proba, columns=iris.target_names)
    st.bar_chart(prob_df.T)

def sentiment_project():
    st.header("😊😡 Sentiment Analyzer")
    st.markdown("""
    **Goal:** Determine if a text is Positive or Negative.
    **Approach:** Simple keyword-based (for demo) + Optional ML.
    """)
    
    st.subheader("1. Try it out!")
    user_text = st.text_area("Enter a review or comment:", "This product is amazing! I love it.")
    
    # Simple keyword-based sentiment (for demo)
    positive_words = ['love', 'amazing', 'great', 'good', 'excellent', 'happy', 'awesome', 'best', 'fantastic', 'wonderful']
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'poor', 'disappointing', 'horrible', 'sad', 'angry']
    
    text_lower = user_text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "Positive 😊"
        color = "green"
    elif neg_count > pos_count:
        sentiment = "Negative 😡"
        color = "red"
    else:
        sentiment = "Neutral 😐"
        color = "gray"
    
    st.markdown(f"### Predicted Sentiment: <span style='color:{color}; font-size:24px;'>{sentiment}</span>", unsafe_allow_html=True)
    
    st.caption(f"Positive keywords found: {pos_count}, Negative keywords found: {neg_count}")
    
    st.info("💡 Real-world sentiment analysis uses trained classifiers like BERT or RoBERTa for much higher accuracy!")

def digit_classifier_project():
    st.header("✍️ Handwritten Digit Classifier")
    st.markdown("""
    **Goal:** Recognize handwritten digits (0-9).
    **Data:** MNIST-like synthetic data.
    """)
    
    st.subheader("🎨 Draw a Digit (Simulated)")
    st.caption("For a real drawing canvas, we'd need a custom component. Here's a simulated demo.")
    
    # Simulated: User picks a digit, we show a random sample
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    
    selected_digit = st.selectbox("Select a digit to see:", list(range(10)))
    
    # Find samples of that digit
    indices = np.where(digits.target == selected_digit)[0]
    sample_idx = np.random.choice(indices)
    
    sample_image = digits.images[sample_idx]
    
    st.write("Sample Image (8x8 pixels):")
    fig = px.imshow(sample_image, color_continuous_scale='gray_r')
    fig.update_layout(width=300, height=300)
    st.plotly_chart(fig)
    
    # Train a simple model (for demo, this is instant)
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', gamma=0.001)
    model.fit(digits.data, digits.target)
    
    prediction = model.predict([digits.data[sample_idx]])[0]
    st.success(f"Model Prediction: **{prediction}**")
    
    st.info("In production, you'd use a CNN (Convolutional Neural Network) for much higher accuracy!")

def recommender_project():
    st.header("🎬 Movie Recommender System")
    st.markdown("""
    **Goal:** Recommend movies based on user preferences.
    **Approach:** Simple Content-Based Filtering using genres.
    """)
    
    # Mock movie data
    movies = pd.DataFrame({
        'title': ['The Matrix', 'Inception', 'Titanic', 'The Godfather', 'Avengers', 'Toy Story', 'The Dark Knight', 'Forrest Gump'],
        'genre': ['Sci-Fi/Action', 'Sci-Fi/Thriller', 'Romance/Drama', 'Crime/Drama', 'Action/Superhero', 'Animation/Comedy', 'Action/Thriller', 'Drama/Romance'],
        'rating': [8.7, 8.8, 7.9, 9.2, 8.0, 8.3, 9.0, 8.8]
    })
    
    st.subheader("1. Movie Database")
    st.dataframe(movies)
    
    st.subheader("2. What do you like?")
    
    liked_movie = st.selectbox("Select a movie you like:", movies['title'].tolist())
    
    # Simple content-based: recommend movies with similar genre
    liked_genre = movies[movies['title'] == liked_movie]['genre'].values[0]
    
    # Find movies with overlapping genres
    def genre_overlap(genre1, genre2):
        set1 = set(genre1.split('/'))
        set2 = set(genre2.split('/'))
        return len(set1.intersection(set2))
    
    movies['similarity'] = movies['genre'].apply(lambda x: genre_overlap(liked_genre, x))
    
    # Exclude the movie itself
    recommendations = movies[movies['title'] != liked_movie].sort_values('similarity', ascending=False).head(3)
    
    st.subheader("3. Recommendations for You")
    
    for idx, row in recommendations.iterrows():
        st.markdown(f"🎬 **{row['title']}** ({row['genre']}) - ⭐ {row['rating']}")
    
    st.info("Real recommender systems use collaborative filtering (user behavior) and matrix factorization for better results!")

