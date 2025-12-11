import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_moons

def show():
    st.title("🕵️ Module 4: Unsupervised Learning")
    
    st.markdown("""
    ### Learning Without a Teacher - Simple + Technical Notes
    """)
    
    tabs = st.tabs(["📚 The Concept", "🧩 Clustering", "📉 Dimensionality Reduction", "📝 Complete Notes", "🎮 Practice"])
    
    # TAB 1: Concept
    with tabs[0]:
        st.header("📚 What is Unsupervised Learning?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **Sorting things without labels!**
            
            Imagine a messy pile of clothes: 👕👖👗🧦
            - No one told you "This is a shirt"
            - But you can group them:
              - Pile 1: Things with sleeves (Shirts)
              - Pile 2: Things for legs (Pants)
              - Pile 3: Tiny things (Socks)
              
            **You find the patterns yourself!**
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Unsupervised Learning** is a type of ML where the 
            model is trained on unlabeled data. The algorithm 
            must discover inherent structures, patterns, or 
            relationships in the input data without explicit 
            output labels.
            
            **Goal:** Find structure in X
            """)
        
        st.markdown("---")
        st.subheader("Two Main Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧩 Clustering
            
            **"Grouping similar things"**
            
            - Grouping customers by shopping habits
            - Grouping news articles by topic
            - Grouping stars by galaxy
            """)
            
        with col2:
            st.markdown("""
            ### 📉 Dimensionality Reduction
            
            **"Simplifying complex things"**
            
            - Making a 3D world look good on a 2D screen
            - Summarizing a long book into a page
            - Compressing data
            """)
    
    # TAB 2: Clustering
    with tabs[1]:
        st.header("🧩 Clustering (Finding Groups)")
        
        st.subheader("🍕 The Pizza Customer Analogy")
        
        st.markdown("""
        You have data on 100 customers:
        1. How much they buy ($)
        2. How often they visit (days)
        
        You want to group them to send coupons.
        """)
        
        # Generate clusters
        X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42)
        df = pd.DataFrame(X, columns=['Spend ($)', 'Frequency (Visits)'])
        
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Renaissance mapping for clusters (Just for demo labels)
        cluster_names = {0: "👑 VIPs", 1: "👻 Occasional", 2: "💸 Savers"}
        # Note: K-Means labels are arbitrary, in real app we'd map based on centroid values
        # For simplicity here, we just use color
        
        fig = px.scatter(df, x='Spend ($)', y='Frequency (Visits)', color=df['Cluster'].astype(str),
                        title="Customer Segments (Groups Found by AI)",
                        color_discrete_sequence=['red', 'blue', 'green'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        ### 💡 AI Found 3 Groups:
        1. **High Spend, High Visits:** (VIPs) → Send "Thank You" card
        2. **Low Spend, Low Visits:** (Occasional) → Send "Come Back" coupon
        3. **Med Spend, Med Visits:** (Regulars) → Send "New Menu" info
        """)
        
        st.markdown("---")
        
        st.subheader("📘 Formal Notes: K-Means")
        
        with st.expander("📖 Algorithm Steps"):
            st.markdown("""
            1. **Initialize:** Pick 'K' random points as centers (centroids).
            2. **Assign:** Assign every data point to the nearest center.
            3. **Update:** Move center to the average (mean) of its points.
            4. **Repeat:** Repeat steps 2-3 until centers stop moving.
            """)
        
        with st.expander("📖 Euclidean Distance Formula"):
            st.markdown("Distance between point P and center C:")
            st.latex(r"d(P, C) = \sqrt{(x_P - x_C)^2 + (y_P - y_C)^2}")
            
        st.subheader("🎮 Interactive K-Means")
        
        k = st.slider("Number of Clusters (K):", 2, 6, 3)
        
        X_cust, _ = make_blobs(n_samples=200, centers=k, cluster_std=1.0, random_state=1)
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(X_cust)
        
        fig2 = px.scatter(x=X_cust[:,0], y=X_cust[:,1], color=labels.astype(str), 
                         title=f"K-Means with K={k}")
        st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 3: Dimensionality Reduction
    with tabs[2]:
        st.header("📉 Dimensionality Reduction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **The Shadow Analogy**
            
            Imagine a floating ball (3D object). 
            If you shine a light, it casts a shadow (2D circle) on the wall.
            
            The shadow is a **simplification** of the ball. 
            It captures the shape without the depth.
            
            **PCA does this with data!** It finds the best angle to cast a "shadow" that keeps the most info.
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Principal Component Analysis (PCA)** is a technique used 
            to reduce the dimensionality of data while preserving as 
            much variance (information) as possible.
            
            It transforms features into new uncorrelated variables 
            called Principal Components.
            """)
            
        st.markdown("---")
        
        st.subheader("🎮 3D to 2D Demo")
        
        # 3D Data
        # Helix
        t = np.linspace(0, 20, 100)
        x = np.cos(t)
        y = np.sin(t)
        z = t
        
        df_3d = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original 3D Data**")
            fig3d = px.scatter_3d(df_3d, x='x', y='y', z='z', title="Complex 3D Helix")
            st.plotly_chart(fig3d, use_container_width=True)
            
        with col2:
            st.markdown("**Flattened 2D Data (PCA)**")
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(df_3d)
            fig2d = px.scatter(x=coords_2d[:,0], y=coords_2d[:,1], title="Simplified 2D Circle(ish)")
            st.plotly_chart(fig2d, use_container_width=True)
            
        st.success("PCA flattened the spring into a circle pattern, keeping the 'loop' structure but losing the height.")

    # TAB 4: Notes
    with tabs[3]:
        st.header("📝 Complete Notes")
        
        st.subheader("🔑 Algorithm Cheat Sheet")
        
        st.markdown("""
        | Algorithm | Type | Use Case | Simple Explanation |
        |-----------|------|----------|--------------------|
        | **K-Means** | Clustering | Customer Segmentation | Finding circular groups of friends |
        | **DBSCAN** | Clustering | Anomaly Detection | Grouping close neighbors, ignoring lonely outliers |
        | **Hierarchical** | Clustering | Taxonomy (Biology) | Building a family tree of groups |
        | **PCA** | Dim. Reduction | Visualization/Compression | Taking the best photo to show maximum detail |
        | **t-SNE/UMAP** | Dim. Reduction | Complex Visualization | Unfolding a crumbled paper ball |
        """)
        
        st.subheader("📚 Key Vocabulary")
        
        vocab = {
            "Cluster": "A group of data points that are similar to each other.",
            "Centroid": "The center point of a cluster.",
            "Outlier": "A data point that doesn't fit into any group (anomaly).",
            "Dimensionality": "The number of features (columns) in your dataset.",
            "Variance": "How spread out the data is. PCA tries to keep this high.",
        }
        
        for term, desc in vocab.items():
            st.markdown(f"**{term}:** {desc}")

    # TAB 5: Practice
    with tabs[4]:
        st.header("🎮 Practice: Be the Algorithm")
        
        st.markdown("**Identify the Clusters!**")
        
        # Generate trickier data (moons)
        X_moon, _ = make_moons(n_samples=200, noise=0.1, random_state=0)
        
        fig = px.scatter(x=X_moon[:,0], y=X_moon[:,1], title="Tricky Data: Concave Shapes")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("If you use **K-Means (Circles)**, what happens?")
        
        if st.button("Run K-Means"):
            km = KMeans(n_clusters=2)
            labels = km.fit_predict(X_moon)
            st.plotly_chart(px.scatter(x=X_moon[:,0], y=X_moon[:,1], color=labels.astype(str), title="K-Means Failed! (Cuts in half)"), use_container_width=True)
            st.error("K-Means likes circles. It fails here!")
            
        st.markdown("If you use **DBSCAN (Density)**, what happens?")
        
        if st.button("Run DBSCAN"):
            dbs = DBSCAN(eps=0.2)
            labels = dbs.fit_predict(X_moon)
            st.plotly_chart(px.scatter(x=X_moon[:,0], y=X_moon[:,1], color=labels.astype(str), title="DBSCAN Worked! (Follows shape)"), use_container_width=True)
            st.success("DBSCAN follows the shape of the data 'snake'! 🐍")
