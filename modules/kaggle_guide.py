import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("🏆 Kaggle Competition Guide")
    
    st.markdown("""
    Master the art of Kaggle competitions! From EDA to winning solutions.
    """)
    
    tabs = st.tabs([
        "📚 Getting Started",
        "🔍 EDA Strategy",
        "🛠️ Feature Engineering",
        "🎯 Model Selection",
        "📊 Ensemble & Stacking",
        "💡 Pro Tips"
    ])
    
    # TAB 1: Getting Started
    with tabs[0]:
        st.header("📚 Getting Started with Kaggle")
        
        st.subheader("What is Kaggle?")
        
        st.markdown("""
        **Kaggle** is the world's largest data science community with:
        - 🏆 Machine learning competitions with prizes
        - 📊 Free datasets
        - 📓 Notebooks (like Jupyter but free GPUs)
        - 📚 Learning courses
        - 🤝 Community discussions
        """)
        
        st.subheader("Competition Types")
        
        st.markdown("""
        | Type | Prize | Duration | Difficulty |
        |------|-------|----------|------------|
        | **Featured** | $10K-$1M+ | 2-3 months | Hard |
        | **Research** | Prestige | Varies | Hard |
        | **Playground** | Swag/Medals | Ongoing | Easy |
        | **Getting Started** | None | Ongoing | Beginner |
        | **Community** | Varies | Varies | Medium |
        """)
        
        st.subheader("The Competition Workflow")
        
        st.graphviz_chart("""
        digraph Workflow {
            rankdir=LR;
            node [shape=box, style=filled];
            
            Join [label="1. Join\\nCompetition", fillcolor=lightyellow];
            Understand [label="2. Understand\\nProblem", fillcolor=lightblue];
            EDA [label="3. EDA &\\nBaseline", fillcolor=lightgreen];
            Features [label="4. Feature\\nEngineering", fillcolor=orange];
            Models [label="5. Model\\nSelection", fillcolor=lightpink];
            Ensemble [label="6. Ensemble\\n& Tune", fillcolor=lavender];
            Submit [label="7. Submit\\n& Iterate", fillcolor=lightyellow];
            
            Join -> Understand -> EDA -> Features -> Models -> Ensemble -> Submit;
            Submit -> Features [style=dashed];
        }
        """)
        
        st.subheader("Starter Code")
        
        st.code("""
# Standard Kaggle imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/competition-name/train.csv')
test = pd.read_csv('/kaggle/input/competition-name/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
train.head()
        """, language="python")
    
    # TAB 2: EDA
    with tabs[1]:
        st.header("🔍 Exploratory Data Analysis (EDA)")
        
        st.markdown("""
        EDA is crucial! Spend 30-40% of your time here.
        """)
        
        st.subheader("EDA Checklist")
        
        checklist = [
            ("📊 Data Overview", "Shape, dtypes, head(), describe(), info()"),
            ("❓ Missing Values", "Heatmap, counts, patterns (MCAR/MAR/MNAR)"),
            ("📈 Target Distribution", "Is it balanced? Skewed? Outliers?"),
            ("🔗 Feature-Target Correlation", "Heatmap, scatter plots"),
            ("📉 Feature Distributions", "Histograms, KDE plots"),
            ("🔢 Numerical Features", "Outliers, scaling needs"),
            ("🏷️ Categorical Features", "Cardinality, rare categories"),
            ("⏰ Time Features", "Trends, seasonality (if applicable)"),
            ("🔍 Train vs Test", "Distribution shift? Leakage?"),
        ]
        
        for emoji_title, desc in checklist:
            st.checkbox(emoji_title, help=desc, key=f"eda_{emoji_title}")
        
        st.subheader("Quick EDA Code")
        
        st.code("""
# 1. Overview
print(train.shape)
print(train.dtypes)
print(train.describe())

# 2. Missing Values
missing = train.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train.isnull(), cbar=True, yticklabels=False)
plt.title("Missing Values Heatmap")

# 3. Target Distribution
sns.histplot(train['target'], kde=True)

# 4. Correlation
corr = train.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)

# 5. Train vs Test Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for col in ['feature1', 'feature2']:
    sns.kdeplot(train[col], label='Train', ax=axes[0])
    sns.kdeplot(test[col], label='Test', ax=axes[0])
plt.legend()
        """, language="python")
    
    # TAB 3: Feature Engineering
    with tabs[2]:
        st.header("🛠️ Feature Engineering")
        
        st.markdown("""
        **Feature Engineering** separates good solutions from great ones.
        
        > "Coming up with features is difficult, time-consuming, requires expert knowledge. 
        > Applied machine learning is basically feature engineering." — Andrew Ng
        """)
        
        st.subheader("Feature Engineering Techniques")
        
        techniques = {
            "🔢 Numerical Features": [
                ("Binning", "Convert continuous to categorical", "pd.cut(df['age'], bins=[0, 18, 65, 100])"),
                ("Log Transform", "Handle skewed data", "np.log1p(df['value'])"),
                ("Square/Cube", "Capture non-linear relationships", "df['feature']**2"),
                ("Interactions", "Multiply features together", "df['f1'] * df['f2']"),
                ("Aggregations", "Group-wise stats", "df.groupby('category')['value'].transform('mean')"),
            ],
            "🏷️ Categorical Features": [
                ("One-Hot Encoding", "Binary columns per category", "pd.get_dummies(df['cat'])"),
                ("Label Encoding", "Integer codes", "LabelEncoder().fit_transform(df['cat'])"),
                ("Target Encoding", "Mean target per category", "df.groupby('cat')['target'].transform('mean')"),
                ("Frequency Encoding", "Count per category", "df['cat'].map(df['cat'].value_counts())"),
                ("Binary Encoding", "Binary representation", "category_encoders.BinaryEncoder()"),
            ],
            "⏰ Time Features": [
                ("Extract Parts", "Year, month, day, hour", "df['date'].dt.year"),
                ("Cyclical Encoding", "Sin/Cos for cyclical", "np.sin(2 * np.pi * df['hour'] / 24)"),
                ("Lag Features", "Previous values", "df['value'].shift(1)"),
                ("Rolling Stats", "Moving averages", "df['value'].rolling(7).mean()"),
                ("Time Since", "Days since event", "(df['date'] - reference_date).dt.days"),
            ],
            "📝 Text Features": [
                ("Length", "Character/word count", "df['text'].str.len()"),
                ("TF-IDF", "Term frequency vectors", "TfidfVectorizer().fit_transform(df['text'])"),
                ("Word Embeddings", "Semantic vectors", "Use pre-trained Word2Vec/BERT"),
                ("Sentiment", "Positive/negative score", "TextBlob(text).sentiment.polarity"),
                ("Entity Extraction", "Named entities", "spacy.load('en').ner"),
            ],
        }
        
        for category, items in techniques.items():
            st.subheader(category)
            for name, desc, code in items:
                with st.expander(f"**{name}**: {desc}"):
                    st.code(code, language="python")
        
        st.subheader("Feature Selection")
        
        st.code("""
# 1. Correlation-based
high_corr = corr[abs(corr['target']) > 0.3].index

# 2. Feature Importance
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

# 3. Recursive Feature Elimination
from sklearn.feature_selection import RFE
selector = RFE(model, n_features_to_select=20)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]

# 4. SHAP Values
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
        """, language="python")
    
    # TAB 4: Model Selection
    with tabs[3]:
        st.header("🎯 Model Selection")
        
        st.subheader("Algorithm Cheat Sheet")
        
        st.markdown("""
        | Problem Type | Baseline | Strong | SOTA |
        |--------------|----------|--------|------|
        | **Binary Classification** | LogisticRegression | XGBoost, LightGBM | CatBoost + Ensemble |
        | **Multi-class Classification** | RandomForest | XGBoost, LightGBM | CatBoost + Neural Nets |
        | **Regression** | Ridge | XGBoost, LightGBM | CatBoost + Ensemble |
        | **Time Series** | ARIMA, Prophet | LightGBM with lags | N-BEATS, Temporal Fusion |
        | **Text Classification** | TF-IDF + LogReg | BERT fine-tuning | DeBERTa, RoBERTa |
        | **Image Classification** | ResNet-18 | EfficientNet | Vision Transformers |
        | **Tabular** | XGBoost | LightGBM, CatBoost | TabNet, SAINT |
        """)
        
        st.subheader("Cross-Validation Strategy")
        
        st.code("""
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified (for imbalanced classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group K-Fold (when samples are grouped, e.g., by user)
gkf = GroupKFold(n_splits=5)

# Time Series Split (for temporal data)
tscv = TimeSeriesSplit(n_splits=5)

# Cross-validation loop
oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
    
    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    test_predictions += model.predict_proba(X_test)[:, 1] / 5

print(f"OOF AUC: {roc_auc_score(y_train, oof_predictions):.4f}")
        """, language="python")
        
        st.subheader("Hyperparameter Tuning")
        
        st.code("""
# Optuna (Recommended)
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
        """, language="python")
    
    # TAB 5: Ensemble
    with tabs[4]:
        st.header("📊 Ensemble & Stacking")
        
        st.markdown("""
        Ensembles combine multiple models for better performance.
        
        > "Ensembles are the key to winning Kaggle competitions" — Every Kaggle Master
        """)
        
        st.subheader("Ensemble Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Simple Averaging")
            st.code("""
# Average predictions
final_pred = (pred1 + pred2 + pred3) / 3

# Weighted averaging
final_pred = 0.5*pred1 + 0.3*pred2 + 0.2*pred3
            """, language="python")
            
        with col2:
            st.markdown("### Voting")
            st.code("""
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model)
], voting='soft')
            """, language="python")
        
        st.subheader("Stacking")
        
        st.graphviz_chart("""
        digraph Stacking {
            rankdir=TB;
            node [shape=box, style=filled];
            
            Data [label="Training Data", fillcolor=lightyellow];
            M1 [label="Model 1\\n(XGBoost)", fillcolor=lightblue];
            M2 [label="Model 2\\n(LightGBM)", fillcolor=lightblue];
            M3 [label="Model 3\\n(CatBoost)", fillcolor=lightblue];
            Meta [label="Meta Model\\n(LogReg/Ridge)", fillcolor=lightgreen];
            Output [label="Final Prediction", fillcolor=lightyellow];
            
            Data -> M1;
            Data -> M2;
            Data -> M3;
            M1 -> Meta;
            M2 -> Meta;
            M3 -> Meta;
            Meta -> Output;
        }
        """)
        
        st.code("""
from sklearn.ensemble import StackingClassifier

# Level 1 models
estimators = [
    ('xgb', XGBClassifier(**xgb_params)),
    ('lgb', LGBMClassifier(**lgb_params)),
    ('cat', CatBoostClassifier(**cat_params)),
    ('rf', RandomForestClassifier(**rf_params)),
]

# Meta model
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True  # Include original features
)

stacking.fit(X_train, y_train)
predictions = stacking.predict_proba(X_test)[:, 1]
        """, language="python")
        
        st.subheader("Blending vs Stacking")
        
        st.markdown("""
        | Aspect | Blending | Stacking |
        |--------|----------|----------|
        | Holdout | Fixed validation set | K-Fold OOF predictions |
        | Data usage | Less efficient | More efficient |
        | Overfitting risk | Lower | Higher (if not careful) |
        | Implementation | Simpler | More complex |
        """)
    
    # TAB 6: Pro Tips
    with tabs[5]:
        st.header("💡 Pro Tips from Kaggle Grandmasters")
        
        tips = [
            ("🎯 Focus on Validation", "A good local validation is more important than public LB. Trust your CV!"),
            ("📊 Spend Time on EDA", "30-40% of your time should be EDA. Understanding data = better features."),
            ("🔧 Simple First", "Start with a simple baseline. Only add complexity when needed."),
            ("📈 Iterate Fast", "Quick experiments > perfect code. Iterate, iterate, iterate."),
            ("🤝 Read Discussions", "The forum is gold. Learn from others' insights and share yours."),
            ("📓 Study Winning Solutions", "After each competition, study the top solutions. Best way to learn."),
            ("🧪 Validate Everything", "Every feature, every hyperparameter change - validate with CV."),
            ("⚠️ Avoid Leakage", "Check for data leakage. If it's too good to be true, it probably is."),
            ("🔀 Diversity in Ensemble", "Diverse models > many similar models. Blend different approaches."),
            ("⏰ Time Management", "Don't overfit to public LB in the last days. Trust your CV."),
        ]
        
        for emoji_title, desc in tips:
            st.markdown(f"### {emoji_title}")
            st.info(desc)
            st.markdown("")
        
        st.subheader("Kaggle Learning Path")
        
        st.markdown("""
        1. **Complete** 3 Getting Started competitions
        2. **Participate** in 5 Playground competitions
        3. **Score** top 10% in any competition (for first medal)
        4. **Read** 10 winning solutions
        5. **Join** a Featured competition and give it your all
        6. **Share** your notebooks and engage with community
        
        **Medals needed for ranks:**
        - 🥉 Kaggle Expert: 2 Bronze (top 40%)
        - 🥈 Kaggle Master: 1 Gold + 2 Silver
        - 🥇 Kaggle Grandmaster: 5 Gold (1 solo), 15+ total medals
        """)
