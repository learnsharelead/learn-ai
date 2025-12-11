import streamlit as st

def show():
    st.title("🎙️ ML Interview Prep")
    
    st.markdown("""
    Ace your Machine Learning interviews with these curated questions and answers!
    """)
    
    tabs = st.tabs(["📚 Fundamentals", "🔢 Algorithms", "🧠 Deep Learning", "💻 Coding", "🎯 System Design", "💡 Tips"])
    
    # TAB 1: Fundamentals
    with tabs[0]:
        st.header("ML Fundamentals Questions")
        
        qa_fundamentals = [
            {
                "q": "What is the difference between Supervised and Unsupervised Learning?",
                "a": """
**Supervised Learning:**
- Uses labeled data (input-output pairs)
- Goal: Learn mapping from X to Y
- Examples: Classification, Regression

**Unsupervised Learning:**
- Uses unlabeled data
- Goal: Find hidden patterns/structure
- Examples: Clustering, Dimensionality Reduction
                """,
                "tip": "Give concrete examples: Email spam (supervised) vs customer segmentation (unsupervised)"
            },
            {
                "q": "What is Overfitting and how do you prevent it?",
                "a": """
**Overfitting:** Model learns training data too well (including noise), fails on new data.

**Signs:**
- High training accuracy, low test accuracy
- Large gap between train and validation loss

**Prevention:**
1. More training data
2. Regularization (L1/L2, Dropout)
3. Cross-validation
4. Early stopping
5. Simpler model (reduce complexity)
6. Data augmentation
                """,
                "tip": "Draw the bias-variance tradeoff curve!"
            },
            {
                "q": "Explain Bias-Variance Tradeoff",
                "a": """
**Bias:** Error from wrong assumptions (underfitting)
- High bias = Model is too simple

**Variance:** Error from sensitivity to training data (overfitting)
- High variance = Model is too complex

**Tradeoff:**
- Simple model: High bias, low variance
- Complex model: Low bias, high variance
- Goal: Find the sweet spot

**Total Error = Bias² + Variance + Irreducible Error**
                """,
                "tip": "Use the dart board analogy: Bias = accuracy, Variance = consistency"
            },
            {
                "q": "What is Regularization?",
                "a": """
**Regularization:** Technique to prevent overfitting by adding penalty for complexity.

**L1 (Lasso):**
- Adds |w| to loss
- Creates sparse weights (feature selection)
- Loss = MSE + λ∑|wᵢ|

**L2 (Ridge):**
- Adds w² to loss
- Shrinks weights towards zero
- Loss = MSE + λ∑wᵢ²

**Elastic Net:** Combination of L1 + L2
                """,
                "tip": "L1 for feature selection, L2 for general regularization"
            },
            {
                "q": "What is Cross-Validation?",
                "a": """
**Cross-Validation:** Technique to evaluate model on multiple train/test splits.

**K-Fold CV:**
1. Split data into K equal parts
2. Train on K-1 parts, test on 1
3. Repeat K times
4. Average the results

**Benefits:**
- More robust evaluation
- Uses all data for training and testing
- Reduces variance of the estimate

**Common K values:** 5 or 10
                """,
                "tip": "Mention stratified K-fold for imbalanced data"
            },
        ]
        
        for i, item in enumerate(qa_fundamentals, 1):
            with st.expander(f"Q{i}: {item['q']}"):
                st.markdown(item['a'])
                st.info(f"💡 **Interview Tip:** {item['tip']}")
    
    # TAB 2: Algorithms
    with tabs[1]:
        st.header("Algorithm Questions")
        
        qa_algorithms = [
            {
                "q": "When would you use Random Forest vs XGBoost?",
                "a": """
**Random Forest:**
- Easy to use, fewer hyperparameters
- Naturally parallel (fast to train)
- Good baseline model
- Less prone to overfitting

**XGBoost (Gradient Boosting):**
- Usually higher accuracy
- Better for competitions
- More hyperparameters to tune
- Handles missing values
- Can overfit if not tuned

**Use Random Forest:** Quick baseline, interpretability needed
**Use XGBoost:** Maximum accuracy, have time to tune
                """,
                "tip": "Mention LightGBM and CatBoost as alternatives to XGBoost"
            },
            {
                "q": "Explain how Decision Trees work",
                "a": """
**Decision Tree:** Tree of if-else rules for prediction.

**Building Process:**
1. Start with all data at root
2. Find best feature/threshold to split
3. Split data into child nodes
4. Repeat recursively until stopping criteria

**Splitting Criteria:**
- Classification: Gini impurity, Information Gain (entropy)
- Regression: MSE reduction

**Stopping Criteria:**
- Max depth reached
- Min samples per leaf
- No improvement in purity
                """,
                "tip": "Be ready to calculate Gini impurity by hand!"
            },
            {
                "q": "How does K-Means work? What are its limitations?",
                "a": """
**K-Means Algorithm:**
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids (mean of assigned points)
4. Repeat 2-3 until convergence

**Limitations:**
- Must choose K beforehand
- Assumes spherical clusters
- Sensitive to initialization
- Sensitive to outliers
- Only finds convex clusters

**Solutions:**
- Elbow method for K selection
- K-Means++ for better initialization
- Use DBSCAN for non-spherical clusters
                """,
                "tip": "Draw non-convex data and show K-Means failure"
            },
            {
                "q": "What is the difference between Bagging and Boosting?",
                "a": """
**Bagging (Bootstrap AGGregatING):**
- Train models in PARALLEL
- Each model on random sample (with replacement)
- Combine by voting/averaging
- Reduces VARIANCE
- Example: Random Forest

**Boosting:**
- Train models SEQUENTIALLY
- Each model focuses on errors of previous
- Combine with weighted voting
- Reduces BIAS
- Example: XGBoost, AdaBoost

**Key difference:** Parallel vs Sequential
                """,
                "tip": "Random Forest = Bagging + Feature randomness"
            },
        ]
        
        for i, item in enumerate(qa_algorithms, 1):
            with st.expander(f"Q{i}: {item['q']}"):
                st.markdown(item['a'])
                st.info(f"💡 **Interview Tip:** {item['tip']}")
    
    # TAB 3: Deep Learning
    with tabs[2]:
        st.header("Deep Learning Questions")
        
        qa_dl = [
            {
                "q": "Explain Backpropagation",
                "a": """
**Backpropagation:** Algorithm to calculate gradients in neural networks.

**Process:**
1. **Forward Pass:** Compute predictions
2. **Calculate Loss:** Compare to ground truth
3. **Backward Pass:** Compute gradients using chain rule
4. **Update Weights:** w = w - lr * gradient

**Chain Rule:**
∂Loss/∂w = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂w

**Why it works:** Efficiently computes all gradients in one backward pass
                """,
                "tip": "Mention vanishing/exploding gradients problem"
            },
            {
                "q": "What is Batch Normalization?",
                "a": """
**Batch Normalization:** Normalize activations within a mini-batch.

**Process:**
1. Compute mean and variance of batch
2. Normalize: x̂ = (x - μ) / √(σ² + ε)
3. Scale and shift: y = γx̂ + β

**Benefits:**
- Faster training
- Higher learning rates possible
- Reduces internal covariate shift
- Acts as regularization

**Where to use:** After linear/conv layer, before activation
                """,
                "tip": "Mention Layer Norm for Transformers, Instance Norm for style transfer"
            },
            {
                "q": "Explain the Transformer architecture",
                "a": """
**Transformer:** Attention-based architecture (no recurrence).

**Key Components:**
1. **Self-Attention:** Each token attends to all others
2. **Multi-Head Attention:** Multiple attention in parallel
3. **Positional Encoding:** Inject position information
4. **Feed-Forward Network:** After attention

**Self-Attention:**
- Q (Query), K (Key), V (Value)
- Attention = softmax(QK^T / √d) V

**Benefits over RNN:**
- Parallelizable (faster)
- Better long-range dependencies
- Foundation for GPT, BERT, etc.
                """,
                "tip": "Draw the attention mechanism and explain Q, K, V"
            },
            {
                "q": "What is Dropout and why does it work?",
                "a": """
**Dropout:** Randomly set neurons to 0 during training.

**Process:**
- During training: Randomly zero out p% of neurons
- During inference: Use all neurons, scale by (1-p)

**Why it works:**
1. Prevents co-adaptation of neurons
2. Ensemble effect (many sub-networks)
3. Acts as regularization
4. Forces redundant representations

**Typical values:** p = 0.2 to 0.5
                """,
                "tip": "Explain why we scale during inference"
            },
        ]
        
        for i, item in enumerate(qa_dl, 1):
            with st.expander(f"Q{i}: {item['q']}"):
                st.markdown(item['a'])
                st.info(f"💡 **Interview Tip:** {item['tip']}")
    
    # TAB 4: Coding
    with tabs[3]:
        st.header("Coding Questions")
        
        st.markdown("Common coding problems in ML interviews:")
        
        coding_qs = [
            {
                "title": "Implement Linear Regression from scratch",
                "code": """
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
                """
            },
            {
                "title": "Implement K-Means from scratch",
                "code": """
import numpy as np

def kmeans(X, k, max_iters=100):
    # Random initialization
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids
                """
            },
            {
                "title": "Implement Softmax function",
                "code": """
import numpy as np

def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Example
logits = np.array([2.0, 1.0, 0.1])
print(softmax(logits))  # [0.659, 0.242, 0.099]
                """
            },
            {
                "title": "Implement Cross-Entropy Loss",
                "code": """
import numpy as np

def cross_entropy(y_true, y_pred, epsilon=1e-15):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary cross-entropy
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                """
            },
        ]
        
        for q in coding_qs:
            with st.expander(q["title"]):
                st.code(q["code"], language="python")
    
    # TAB 5: System Design
    with tabs[4]:
        st.header("ML System Design Questions")
        
        st.markdown("""
        These questions evaluate your ability to design end-to-end ML systems.
        """)
        
        design_qs = [
            {
                "q": "Design a recommendation system for Netflix",
                "framework": """
**1. Clarify Requirements:**
- Scale: 200M users, 15K titles
- Latency: < 200ms
- Metrics: Engagement, watch time

**2. Data:**
- User behavior (watch history, ratings)
- Content metadata (genre, actors)
- Context (time, device)

**3. Approach:**
- Collaborative Filtering (user-item matrix)
- Content-Based (similar content)
- Hybrid (combine both)
- Deep Learning (two-tower model)

**4. System Design:**
- Offline: Train models, generate embeddings
- Near-realtime: Update with recent behavior
- Online: Retrieve candidates, rank, serve

**5. Evaluation:**
- Offline: Precision@K, NDCG
- Online: A/B test (watch time, CTR)
                """
            },
            {
                "q": "Design a fraud detection system",
                "framework": """
**1. Requirements:**
- Real-time detection (< 100ms)
- High recall (catch most fraud)
- Handle class imbalance (99.9% legitimate)

**2. Features:**
- Transaction amount, location, time
- User behavior patterns
- Device fingerprint
- Network features (graph)

**3. Model:**
- Gradient Boosting for tabular features
- Graph neural networks for networks
- Anomaly detection (Isolation Forest)
- Ensemble of models

**4. Handling Imbalance:**
- SMOTE / undersampling
- Weighted loss
- Anomaly detection approach

**5. System:**
- Real-time scoring pipeline
- Rule engine for known patterns
- Human review queue for edge cases
- Feedback loop for labels
                """
            },
        ]
        
        for item in design_qs:
            with st.expander(item["q"]):
                st.markdown(item["framework"])
    
    # TAB 6: Tips
    with tabs[5]:
        st.header("💡 Interview Tips")
        
        st.markdown("""
        ### Before the Interview
        
        1. **Review Fundamentals:** Bias-variance, regularization, metrics
        2. **Practice Coding:** Implement algorithms from scratch
        3. **Study System Design:** End-to-end ML pipelines
        4. **Prepare Projects:** Know your resume deeply
        5. **Mock Interviews:** Practice with peers
        
        ### During the Interview
        
        1. **Think Out Loud:** Explain your reasoning
        2. **Ask Clarifying Questions:** Don't assume
        3. **Start Simple:** Baseline before complex
        4. **Discuss Trade-offs:** Show you understand nuances
        5. **Be Honest:** Say "I don't know" if needed
        
        ### Common Mistakes to Avoid
        
        ❌ Jumping to complex solutions
        ❌ Ignoring edge cases
        ❌ Not asking about the data
        ❌ Forgetting to evaluate
        ❌ Not considering production constraints
        
        ### Resources
        
        - 📖 "Machine Learning Interviews" book
        - 🎥 YouTube: StatQuest, 3Blue1Brown
        - 💻 LeetCode ML section
        - 📝 Chip Huyen's ML Interview Book (free online)
        """)
