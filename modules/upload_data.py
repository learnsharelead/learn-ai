import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

def show():
    st.title("📥 Upload Your Data")
    
    st.markdown("""
    Upload your own CSV file and let AI analyze it! Build and train models on your data.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Store in session
        st.session_state.user_data = df
        
        # Tabs for different operations
        tabs = st.tabs(["📊 Explore", "🧹 Preprocess", "🤖 Train Model", "📈 Visualize"])
        
        # TAB 1: Explore
        with tabs[0]:
            st.header("Data Exploration")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.subheader("Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Column Info")
            col_info = pd.DataFrame({
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.subheader("Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        # TAB 2: Preprocess
        with tabs[1]:
            st.header("Data Preprocessing")
            
            st.subheader("Handle Missing Values")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                st.warning(f"Columns with missing values: {missing_cols}")
                
                strategy = st.selectbox("Missing Value Strategy:", 
                    ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with 0"])
                
                if st.button("Apply Missing Value Strategy"):
                    if strategy == "Drop Rows":
                        df = df.dropna()
                    elif strategy == "Fill with Mean":
                        df = df.fillna(df.mean(numeric_only=True))
                    elif strategy == "Fill with Median":
                        df = df.fillna(df.median(numeric_only=True))
                    elif strategy == "Fill with Mode":
                        df = df.fillna(df.mode().iloc[0])
                    elif strategy == "Fill with 0":
                        df = df.fillna(0)
                    
                    st.session_state.user_data = df
                    st.success(f"Applied! New shape: {df.shape}")
                    st.rerun()
            else:
                st.success("✅ No missing values!")
            
            st.subheader("Feature Selection")
            
            selected_cols = st.multiselect("Select columns to keep:", df.columns.tolist(), default=df.columns.tolist())
            
            if st.button("Apply Column Selection"):
                df = df[selected_cols]
                st.session_state.user_data = df
                st.success(f"Kept {len(selected_cols)} columns")
                st.rerun()
        
        # TAB 3: Train Model
        with tabs[2]:
            st.header("Train a Model")
            
            df = st.session_state.user_data
            
            # Select problem type
            problem_type = st.radio("Problem Type:", ["Classification", "Regression", "Clustering"])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            
            if problem_type in ["Classification", "Regression"]:
                st.subheader("Select Features & Target")
                
                target_col = st.selectbox("Target Column (y):", all_cols)
                feature_cols = st.multiselect("Feature Columns (X):", 
                    [c for c in numeric_cols if c != target_col],
                    default=[c for c in numeric_cols if c != target_col][:5])
                
                if st.button("🚀 Train Model"):
                    if not feature_cols:
                        st.error("Please select at least one feature!")
                    else:
                        with st.spinner("Training..."):
                            X = df[feature_cols].dropna()
                            y = df.loc[X.index, target_col]
                            
                            # Encode if categorical
                            if y.dtype == 'object':
                                le = LabelEncoder()
                                y = le.fit_transform(y)
                            
                            # Split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Scale
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            if problem_type == "Classification":
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                score = accuracy_score(y_test, y_pred)
                                metric_name = "Accuracy"
                            else:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                score = r2_score(y_test, y_pred)
                                metric_name = "R² Score"
                            
                            st.success(f"✅ Model Trained! {metric_name}: {score:.4f}")
                            
                            # Feature Importance
                            st.subheader("Feature Importance")
                            importance = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Predictions vs Actual
                            st.subheader("Predictions vs Actual")
                            
                            if problem_type == "Regression":
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
                                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                    mode='lines', name='Perfect Fit', line=dict(dash='dash')))
                                fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                from sklearn.metrics import confusion_matrix
                                cm = confusion_matrix(y_test, y_pred)
                                fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
                                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Clustering
                st.subheader("Select Features for Clustering")
                
                feature_cols = st.multiselect("Feature Columns:", numeric_cols, default=numeric_cols[:2])
                n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
                
                if st.button("🚀 Run Clustering"):
                    if len(feature_cols) < 2:
                        st.error("Please select at least 2 features!")
                    else:
                        X = df[feature_cols].dropna()
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(X_scaled)
                        
                        sil_score = silhouette_score(X_scaled, labels)
                        st.success(f"✅ Clustering Complete! Silhouette Score: {sil_score:.4f}")
                        
                        # Visualize
                        if len(feature_cols) >= 2:
                            fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=labels.astype(str),
                                labels={'x': feature_cols[0], 'y': feature_cols[1], 'color': 'Cluster'})
                            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 4: Visualize
        with tabs[3]:
            st.header("Data Visualization")
            
            df = st.session_state.user_data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            viz_type = st.selectbox("Visualization Type:", 
                ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap", "Pair Plot"])
            
            if viz_type == "Histogram":
                col = st.selectbox("Column:", numeric_cols)
                fig = px.histogram(df, x=col, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                color_col = st.selectbox("Color by (optional):", ["None"] + df.columns.tolist())
                
                if color_col == "None":
                    fig = px.scatter(df, x=x_col, y=y_col)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Box Plot":
                col = st.selectbox("Column:", numeric_cols)
                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Correlation Heatmap":
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto='.2f', labels=dict(color="Correlation"))
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Pair Plot":
                if len(numeric_cols) > 5:
                    st.warning("Too many columns. Selecting first 5.")
                    cols_to_plot = numeric_cols[:5]
                else:
                    cols_to_plot = numeric_cols
                    
                fig = px.scatter_matrix(df[cols_to_plot])
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Sample datasets
        st.info("👆 Upload your own CSV file, or try one of our sample datasets below:")
        
        sample_datasets = {
            "Iris (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Tips (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "Boston Housing (Regression)": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        }
        
        col1, col2 = st.columns(2)
        
        for i, (name, url) in enumerate(sample_datasets.items()):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"📊 Load {name}", key=f"sample_{i}"):
                    st.session_state.sample_url = url
                    st.experimental_set_query_params(sample=name)
                    st.rerun()
        
        # Load sample if selected
        if 'sample_url' in st.session_state:
            df = pd.read_csv(st.session_state.sample_url)
            st.session_state.user_data = df
            st.success(f"Loaded sample dataset: {df.shape[0]} rows x {df.shape[1]} columns")
            st.dataframe(df.head())
