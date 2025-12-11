import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def show():
    st.header("⚔️ Model Arena")
    st.markdown("Compare different AI algorithms side-by-side on synthetic datasets. Visualize how they learn decision boundaries!")
    
    # --- Sidebar Controls ---
    col_controls, col_viz = st.columns([1, 3])
    
    with col_controls:
        st.subheader("⚙️ Configuration")
        
        # Dataset Selection
        dataset_type = st.selectbox(
            "1. Choose Dataset",
            ["Moons", "Circles", "Linearly Separable"]
        )
        
        noise = st.slider("Noise Level", 0.0, 0.5, 0.2, step=0.05)
        n_samples = st.slider("Sample Size", 100, 1000, 300, step=50)
        
        st.markdown("---")
        
        # Model 1 Selection
        st.subheader("🥊 Red Corner")
        model1_name = st.selectbox("Select Model A", ["KNN", "Linear SVM", "RBF SVM", "Random Forest", "Neural Net"], index=0)
        
        st.markdown("---")
        
        # Model 2 Selection
        st.subheader("🥊 Blue Corner")
        model2_name = st.selectbox("Select Model B", ["KNN", "Linear SVM", "RBF SVM", "Random Forest", "Neural Net"], index=4)
        
        btn_run = st.button("🚀 Fight!", use_container_width=True)

    # --- Data Generation ---
    X, y = None, None
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   random_state=42, n_clusters_per_class=1)
        X += 2 * np.random.uniform(size=X.shape)
        
    # Scale Data
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # --- Plotting Helper ---
    def get_mesh_grid(X, h=.02):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def train_and_predict(model_name, X_train, y_train, xx, yy):
        clf = None
        if model_name == "KNN":
            clf = KNeighborsClassifier(3)
        elif model_name == "Linear SVM":
            clf = SVC(kernel="linear", C=0.025, probability=True)
        elif model_name == "RBF SVM":
            clf = SVC(gamma=2, C=1, probability=True)
        elif model_name == "Random Forest":
            clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        elif model_name == "Neural Net":
            clf = MLPClassifier(alpha=1, max_iter=1000)
        elif model_name == "Naive Bayes":
            clf = GaussianNB()
            
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
        return Z.reshape(xx.shape), score

    # --- Visualisation ---
    with col_viz:
        if btn_run:
            xx, yy = get_mesh_grid(X)
            
            c1, c2 = st.columns(2)
            
            # --- Model 1 ---
            with c1:
                with st.spinner(f"Training {model1_name}..."):
                    Z1, score1 = train_and_predict(model1_name, X_train, y_train, xx, yy)
                
                st.markdown(f"### {model1_name}")
                st.metric("Test Accuracy", f"{score1:.2%}")
                
                fig1 = go.Figure(data=[
                    go.Contour(
                        x=np.arange(xx.min(), xx.max(), 0.02),
                        y=np.arange(yy.min(), yy.max(), 0.02),
                        z=Z1,
                        colorscale="RdBu",
                        opacity=0.8,
                        showscale=False
                    ),
                    go.Scatter(
                        x=X_test[:, 0], y=X_test[:, 1],
                        mode='markers',
                        marker=dict(color=y_test, colorscale="RdBu", line_width=1, line_color='white')
                    )
                ])
                fig1.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig1, use_container_width=True)

            # --- Model 2 ---
            with c2:
                with st.spinner(f"Training {model2_name}..."):
                    Z2, score2 = train_and_predict(model2_name, X_train, y_train, xx, yy)
                
                st.markdown(f"### {model2_name}")
                st.metric("Test Accuracy", f"{score2:.2%}", delta=f"{(score2-score1):.2%}")
                
                fig2 = go.Figure(data=[
                    go.Contour(
                        x=np.arange(xx.min(), xx.max(), 0.02),
                        y=np.arange(yy.min(), yy.max(), 0.02),
                        z=Z2,
                        colorscale="RdBu",
                        opacity=0.8,
                        showscale=False
                    ),
                    go.Scatter(
                        x=X_test[:, 0], y=X_test[:, 1],
                        mode='markers',
                        marker=dict(color=y_test, colorscale="RdBu", line_width=1, line_color='white')
                    )
                ])
                fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig2, use_container_width=True)

        else:
            # Initial Empty State
            st.info("👈 Select configuration and click 'Fight!' to compare models.")
            
            # Show Raw Data Preview
            fig_raw = px.scatter(x=X[:, 0], y=X[:, 1], color=y, 
                               title=f"Sample Dataset: {dataset_type}",
                               color_continuous_scale="RdBu")
            st.plotly_chart(fig_raw, use_container_width=True)

