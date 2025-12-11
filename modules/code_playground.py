import streamlit as st
from streamlit_ace import st_ace
import sys
from io import StringIO
import traceback

def show():
    st.title("🔬 Live Code Playground")
    
    st.markdown("""
    Write and run Python code in real-time! Experiment with ML concepts.
    """)
    
    # Template selection
    templates = {
        "Empty": "",
        "Linear Regression": '''import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(50, 1) * 10
y = 2 * X + 3 + np.random.randn(50, 1)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Print results
print(f"Slope (coefficient): {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"R² Score: {model.score(X, y):.4f}")
''',
        "K-Means Clustering": '''import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Print results
print(f"Cluster Centers:\\n{kmeans.cluster_centers_}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Number of points per cluster: {np.bincount(y_pred)}")
''',
        "Neural Network (MLP)": '''import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate
train_acc = mlp.score(X_train, y_train)
test_acc = mlp.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Number of layers: {mlp.n_layers_}")
print(f"Number of iterations: {mlp.n_iter_}")
''',
        "Text Vectorization": '''from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is awesome",
    "Deep learning uses neural networks",
    "I love neural networks"
]

# Create Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Display vocabulary
print("Vocabulary:", vectorizer.get_feature_names_out())
print("\\nDocument-Term Matrix:")
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df)
''',
        "Decision Tree": '''from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Results
print(f"Accuracy: {tree.score(X_test, y_test):.2%}")
print(f"Feature Importances:")
for name, importance in zip(iris.feature_names, tree.feature_importances_):
    print(f"  {name}: {importance:.3f}")
''',
        "Cross-Validation": '''from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-Validation Scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.2%}")
print(f"Standard Deviation: {np.std(scores):.2%}")
'''
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        template = st.selectbox("📋 Load Template:", list(templates.keys()))
        
        st.markdown("### ⚙️ Settings")
        theme = st.selectbox("Theme:", ["monokai", "github", "tomorrow", "twilight", "solarized_dark"])
        font_size = st.slider("Font Size:", 12, 24, 14)
    
    with col1:
        st.markdown("### ✏️ Write Your Code")
        
        # Get initial code
        if 'playground_code' not in st.session_state:
            st.session_state.playground_code = templates[template]
        
        if template != "Empty" and st.button("📥 Load Template"):
            st.session_state.playground_code = templates[template]
            st.rerun()
        
        # Code editor
        code = st_ace(
            value=st.session_state.playground_code,
            language="python",
            theme=theme,
            font_size=font_size,
            height=400,
            key="ace_editor"
        )
        
        st.session_state.playground_code = code
    
    # Run button
    col_run, col_clear = st.columns([1, 1])
    
    with col_run:
        run_button = st.button("▶️ Run Code", type="primary", use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear Output", use_container_width=True):
            if 'playground_output' in st.session_state:
                del st.session_state.playground_output
            st.rerun()
    
    # Execute code
    if run_button and code:
        st.markdown("### 📤 Output")
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            # Create a restricted namespace
            namespace = {
                '__builtins__': __builtins__,
                'np': __import__('numpy'),
                'pd': __import__('pandas'),
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Get output
            output = mystdout.getvalue()
            
            if output:
                st.code(output, language="text")
            else:
                st.info("Code executed successfully (no print output)")
            
            st.session_state.playground_output = output
            
        except Exception as e:
            st.error("❌ Error:")
            st.code(traceback.format_exc(), language="text")
        
        finally:
            sys.stdout = old_stdout
    
    # Tips
    st.markdown("---")
    st.subheader("💡 Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Available Libraries:**
        - `numpy` (as `np`)
        - `pandas` (as `pd`)
        - `sklearn` (scikit-learn)
        - `matplotlib.pyplot` (as `plt`)
        """)
    
    with col2:
        st.markdown("""
        **Best Practices:**
        - Use `print()` to see outputs
        - Start with templates to learn
        - Modify parameters and observe changes
        - Experiment freely!
        """)
    
    # Challenges
    st.markdown("---")
    st.subheader("🎯 Coding Challenges")
    
    challenges = [
        {
            "title": "Challenge 1: Modify Linear Regression",
            "desc": "Change the slope from 2 to 5 and observe how the model learns the new relationship.",
            "hint": "Modify the line `y = 2 * X + 3 + ...`"
        },
        {
            "title": "Challenge 2: Change K in K-Means",
            "desc": "What happens when you set n_clusters to 2 instead of 4? What about 8?",
            "hint": "Modify `n_clusters=4` in KMeans"
        },
        {
            "title": "Challenge 3: Neural Network Depth",
            "desc": "Try changing `hidden_layer_sizes=(10, 10)` to `(50, 50, 50)`. Does accuracy improve?",
            "hint": "More layers = deeper network"
        }
    ]
    
    for i, challenge in enumerate(challenges):
        with st.expander(challenge["title"]):
            st.write(challenge["desc"])
            st.caption(f"💡 Hint: {challenge['hint']}")
