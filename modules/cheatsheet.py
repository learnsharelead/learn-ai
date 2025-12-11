import streamlit as st

def show():
    st.title("📋 AI/ML Cheat Sheet")
    
    st.markdown("""
    A quick reference for algorithms, formulas, and when to use what.
    """)
    
    tabs = st.tabs(["🎯 Algorithm Selector", "📐 Formulas", "🔧 Sklearn Quick Reference", "📚 Glossary"])
    
    # TAB 1: Algorithm Selector
    with tabs[0]:
        st.header("Which Algorithm Should I Use?")
        
        st.subheader("🤔 Answer These Questions:")
        
        q1 = st.radio("Do you have labeled data (target variable)?", ["Yes (Supervised)", "No (Unsupervised)"])
        
        if q1 == "Yes (Supervised)":
            q2 = st.radio("What type of target?", ["Continuous (Numbers)", "Categorical (Classes)"])
            
            if q2 == "Continuous (Numbers)":
                st.success("### ✅ Use REGRESSION")
                st.markdown("""
                | Algorithm | When to Use | Pros | Cons |
                |-----------|-------------|------|------|
                | **Linear Regression** | Simple relationship | Fast, interpretable | Assumes linearity |
                | **Ridge/Lasso** | Many features, regularization needed | Prevents overfitting | Requires tuning |
                | **Random Forest Regressor** | Non-linear, complex data | Robust, feature importance | Slower, less interpretable |
                | **XGBoost** | Competition/Production | Best accuracy | Complex, overfits easily |
                """)
            else:
                st.success("### ✅ Use CLASSIFICATION")
                st.markdown("""
                | Algorithm | When to Use | Pros | Cons |
                |-----------|-------------|------|------|
                | **Logistic Regression** | Binary, need probabilities | Fast, interpretable | Linear boundary only |
                | **Decision Tree** | Need interpretability | Visual, no scaling needed | Overfits easily |
                | **Random Forest** | General purpose | Robust, handles noise | Slower, black box |
                | **SVM** | High dimensional data | Works well with few samples | Slow on large data |
                | **Neural Network** | Images, text, complex patterns | Most powerful | Needs lots of data |
                """)
        else:
            q3 = st.radio("What's your goal?", ["Find Groups (Clustering)", "Reduce Dimensions", "Find Anomalies"])
            
            if q3 == "Find Groups (Clustering)":
                st.success("### ✅ Use CLUSTERING")
                st.markdown("""
                | Algorithm | When to Use | Pros | Cons |
                |-----------|-------------|------|------|
                | **K-Means** | Round clusters, known k | Fast, simple | Need to specify k |
                | **DBSCAN** | Arbitrary shapes, outliers | No k needed, finds outliers | Sensitive to eps |
                | **Hierarchical** | Want a dendrogram | Visual hierarchy | Slow on large data |
                """)
            elif q3 == "Reduce Dimensions":
                st.success("### ✅ Use DIMENSIONALITY REDUCTION")
                st.markdown("""
                | Algorithm | When to Use |
                |-----------|-------------|
                | **PCA** | Linear relationships, preprocessing |
                | **t-SNE** | Visualization (2D/3D) |
                | **UMAP** | Faster t-SNE alternative |
                """)
            else:
                st.success("### ✅ Use ANOMALY DETECTION")
                st.markdown("""
                | Algorithm | When to Use |
                |-----------|-------------|
                | **Isolation Forest** | General purpose |
                | **One-Class SVM** | High dimensional |
                | **DBSCAN** | Density-based outliers |
                """)

    # TAB 2: Formulas
    with tabs[1]:
        st.header("📐 Key Formulas")
        
        st.subheader("Regression Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"MSE = \frac{1}{n}\sum(y - \hat{y})^2")
            st.latex(r"RMSE = \sqrt{MSE}")
        with col2:
            st.latex(r"MAE = \frac{1}{n}\sum|y - \hat{y}|")
            st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
        
        st.subheader("Classification Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}")
            st.latex(r"Precision = \frac{TP}{TP + FP}")
        with col2:
            st.latex(r"Recall = \frac{TP}{TP + FN}")
            st.latex(r"F1 = 2 \times \frac{P \times R}{P + R}")
        
        st.subheader("Neural Networks")
        st.latex(r"Sigmoid: \sigma(x) = \frac{1}{1 + e^{-x}}")
        st.latex(r"ReLU: f(x) = max(0, x)")
        st.latex(r"Softmax: P(y=k) = \frac{e^{z_k}}{\sum_j e^{z_j}}")
        
        st.subheader("Loss Functions")
        st.latex(r"MSE Loss: L = \frac{1}{n}\sum(y - \hat{y})^2")
        st.latex(r"Cross-Entropy: L = -\sum y \log(\hat{y})")

    # TAB 3: Sklearn Quick Reference
    with tabs[2]:
        st.header("🔧 Scikit-Learn Quick Reference")
        
        st.code("""
# ===== DATA PREPROCESSING =====
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===== REGRESSION =====
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# ===== CLASSIFICATION =====
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)


# ===== CLUSTERING =====
from sklearn.cluster import KMeans, DBSCAN

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)


# ===== METRICS =====
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
        """, language="python")

    # TAB 4: Glossary
    with tabs[3]:
        st.header("📚 Glossary")
        
        terms = {
            "Epoch": "One complete pass through the entire training dataset.",
            "Batch Size": "Number of samples processed before updating model weights.",
            "Learning Rate": "Step size for weight updates. Too high = unstable, too low = slow.",
            "Overfitting": "Model memorizes training data, fails on new data.",
            "Underfitting": "Model is too simple to capture patterns.",
            "Regularization": "Technique to prevent overfitting (L1, L2, Dropout).",
            "Gradient Descent": "Optimization algorithm that minimizes loss by following gradients.",
            "Backpropagation": "Algorithm to calculate gradients in neural networks.",
            "Feature Engineering": "Creating new features from existing data.",
            "Hyperparameter": "Settings you choose before training (learning rate, layers, etc.).",
            "Cross-Validation": "Splitting data multiple ways to get robust evaluation.",
            "Embedding": "Dense vector representation of discrete items (words, users, etc.).",
            "Transformer": "Architecture using self-attention. Basis for GPT, BERT.",
            "Fine-Tuning": "Training a pre-trained model on your specific data.",
            "RAG": "Retrieval-Augmented Generation. LLM + external knowledge base.",
            "Hallucination": "When LLMs generate false or made-up information.",
            "Token": "A piece of text (word or subword) that LLMs process.",
            "Inference": "Using a trained model to make predictions.",
            "MLOps": "Practices for deploying and maintaining ML in production."
        }
        
        search = st.text_input("🔍 Search terms:")
        
        for term, definition in terms.items():
            if search.lower() in term.lower() or search.lower() in definition.lower() or search == "":
                st.markdown(f"**{term}**: {definition}")
