import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import plotly.express as px

def show():
    st.title("🗣️ Natural Language Processing (NLP)")
    
    st.markdown("""
    **NLP** bridges the gap between computers and human language.
    Machines only understand numbers, so how do we feed them Shakespeare?
    """)
    
    tab1, tab2 = st.tabs(["Bag of Words (BoW)", "Word Embeddings"])
    
    with tab1:
        st.header("The Simplest Model: Bag of Words")
        st.write("We simply count how many times each word appears. Order doesn't matter.")
        
        st.subheader("Try it out!")
        sentence_1 = st.text_input("Sentence 1:", "The cat sat on the mat")
        sentence_2 = st.text_input("Sentence 2:", "The dog sat on the log")
        sentence_3 = st.text_input("Sentence 3:", "The cat chases the dog")
        
        corpus = [sentence_1, sentence_2, sentence_3]
        
        if st.button("Vectorize Text"):
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            df_bow = pd.DataFrame(X.toarray(), columns=feature_names, index=["S1", "S2", "S3"])
            
            st.write("### Document-Term Matrix")
            st.dataframe(df_bow.style.highlight_max(axis=0))
            
            st.write("Each sentence is now a vector of numbers!")
            
    with tab2:
        st.header("Word Embeddings (Advanced)")
        st.write("Bag of Words is simple, but it doesn't understand *meaning*. 'Cat' and 'Dog' are just different indices.")
        st.write("**Embeddings** map words to a dense vector space where similar words are closer together.")
        
        st.caption("Simplified 3D visualization of Word2Vec concept:")
        
        # Mock 3D data for visualization
        data = {
            'word': ['King', 'Queen', 'Man', 'Woman', 'Apple', 'Banana', 'Orange'],
            'x': [5, 5, 4, 4, 1, 1, 1],
            'y': [1, 1, 1, 1, 5, 5, 5],
            'z': [5, 4, 5, 4, 2, 2, 2],
            'category': ['Royalty', 'Royalty', 'Gender', 'Gender', 'Fruit', 'Fruit', 'Fruit']
        }
        
        fig = px.scatter_3d(data, x='x', y='y', z='z', text='word', color='category')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Notice the relationships:**
        - **King** is close to **Queen** (Royalty cluster).
        - **King - Man + Woman ≈ Queen** (Vector arithmetic!).
        """)
