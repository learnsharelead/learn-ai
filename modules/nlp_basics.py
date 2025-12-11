import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import plotly.express as px
import numpy as np
import altair as alt

def show():
    st.title("🗣️ Natural Language Processing (NLP)")
    
    st.markdown("""
    ### Teaching Machines to Read
    
    **NLP** is how computers understand, interpret, and generate human language.
    From simple word counts to complex Transformer models.
    """)
    
    tabs = st.tabs([
        "📜 Text Preprocessing",
        "🧮 Bag of Words & TF-IDF",
        "🔢 Word Embeddings",
        "🎭 Sentiment Analysis",
        "🧠 Transformers (Attention)"
    ])
    
    # TAB 1: Preprocessing
    with tabs[0]:
        st.header("📜 Step 1: Cleaning Text")
        
        st.markdown("""
        Raw text is messy. Before AI can read it, we must clean it.
        
        **Common Steps:**
        1. **Lowercasing:** "Apple" == "apple"
        2. **Tokenization:** Splitting sentences into words.
        3. **Stop Word Removal:** Removing common words ("the", "is", "at").
        4. **Stemming/Lemmatization:** "Running" -> "Run".
        """)
        
        st.subheader("Interactive Preprocessing")
        
        raw_text = st.text_area("Enter Text:", "The quick brown foxes are jumping over the lazy dog! #Amazing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Tokens:**")
            st.code(raw_text.split())
            
        with col2:
            st.markdown("**Cleaned Tokens:**")
            # Simple cleaning simulation
            import re
            clean = re.sub(r'[^a-zA-Z\s]', '', raw_text).lower()
            tokens = clean.split()
            stopwords = {"the", "is", "at", "which", "on", "are"}
            filtered = [w for w in tokens if w not in stopwords]
            st.code(filtered)
            
        st.info("Notice how punctuation is gone, case is lowered, and 'the' is removed.")
    
    # TAB 2: BoW & TF-IDF
    with tabs[1]:
        st.header("🧮 Turning Text into Numbers")
        
        mode = st.radio("Technique:", ["Count Vectorizer (Bag of Words)", "TF-IDF"])
        
        st.subheader("Interactive Demo")
        s1 = st.text_input("Doc 1:", "The cat sat on the mat")
        s2 = st.text_input("Doc 2:", "The dog sat on the log")
        s3 = st.text_input("Doc 3:", "The cat chases the dog")
        
        corpus = [s1, s2, s3]
        
        if mode == "Count Vectorizer (Bag of Words)":
            vectorizer = CountVectorizer()
            st.caption("Counts how many times each word appears.")
        else:
            vectorizer = TfidfVectorizer()
            st.caption("**TF-IDF** lowers the weight of common words (like 'the') and highlights unique ones.")
            
        X = vectorizer.fit_transform(corpus)
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=["Doc 1", "Doc 2", "Doc 3"])
        
        st.dataframe(df.style.background_gradient(cmap="Blues", axis=None))
        
        st.subheader("Cosine Similarity")
        st.write("How similar are these documents?")
        sim_matrix = cosine_similarity(X)
        st.write(pd.DataFrame(sim_matrix, columns=["Doc 1", "Doc 2", "Doc 3"], index=["Doc 1", "Doc 2", "Doc 3"]))

    # TAB 3: Embeddings
    with tabs[2]:
        st.header("🔢 Word Embeddings (Word2Vec)")
        
        st.markdown("""
        **Bag of Words** has a problem: It thinks "Cat" and "Dog" are totally different.
        
        **Embeddings** map words to a vector space where *meaning* is preserved.
        - "King" - "Man" + "Woman" ≈ "Queen"
        """)
        
        # 3D Visualization
        data = {
            'word': ['King', 'Queen', 'Man', 'Woman', 'Apple', 'Banana', 'Orange', 'Car', 'Bus'],
            'x': [5, 5, 4, 4, 1, 1, 1, 8, 8],
            'y': [1, 1, 1, 1, 5, 5, 5, 2, 2],
            'z': [5, 4, 5, 4, 2, 2, 2, 8, 9],
            'category': ['Royalty', 'Royalty', 'Gender', 'Gender', 'Fruit', 'Fruit', 'Fruit', 'Vehicle', 'Vehicle']
        }
        fig = px.scatter_3d(data, x='x', y='y', z='z', text='word', color='category')
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Sentiment Analysis
    with tabs[3]:
        st.header("🎭 Sentiment Analysis Demo")
        
        st.markdown("A simple classifier trained to detect **Positive** vs **Negative** vibes.")
        
        # Training data (micro dataset)
        train_data = [
            ("I love this movie", "Positive"),
            ("This is amazing", "Positive"),
            ("Best experience ever", "Positive"),
            ("I hate this", "Negative"),
            ("Terrible service", "Negative"),
            ("Waste of time", "Negative")
        ]
        
        # Train model live
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        X_train = [x[0] for x in train_data]
        y_train = [x[1] for x in train_data]
        model.fit(X_train, y_train)
        
        user_text = st.text_input("Test the AI:", "I really enjoyed the service")
        
        if user_text:
            pred = model.predict([user_text])[0]
            probs = model.predict_proba([user_text])[0]
            
            col1, col2 = st.columns(2)
            with col1:
                if pred == "Positive":
                    st.success(f"**Prediction: {pred}** 🎉")
                else:
                    st.error(f"**Prediction: {pred}** 😠")
            
            with col2:
                st.write(f"Confidence: {max(probs):.1%}")

    # TAB 5: Attention Visualizer
    with tabs[4]:
        st.header("🧠 The Transformer Revolution")
        
        st.markdown("""
        **Attention is All You Need (2017)** changed everything.
        Instead of reading left-to-right, Transformers look at the whole sentence at once using **Self-Attention**.
        """)
        
        st.subheader("Visualizing Attention Scores")
        st.write("See how the word **'bank'** pays attention to context to know its meaning.")
        
        sentence = ["The", "river", "bank", "was", "muddy"]
        
        # Mock attention weights
        # 'bank' should pay high attention to 'river'
        attention_weights = [0.05, 0.60, 0.20, 0.10, 0.05]
        
        df_att = pd.DataFrame({"Token": sentence, "Attention": attention_weights})
        
        c = alt.Chart(df_att).mark_bar().encode(
            x='Token',
            y='Attention',
            color=alt.condition(
                alt.datum.Attention > 0.5,
                alt.value('orange'),  # The highlight
                alt.value('steelblue')
            )
        ).properties(title="Attention weights for word: 'bank'")
        
        st.altair_chart(c, use_container_width=True)
        
        st.info("The high attention to 'river' tells the AI that 'bank' means **river bank**, not **money bank**.")
