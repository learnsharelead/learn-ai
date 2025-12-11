import streamlit as st
from modules import (
    # Core Learning Modules
    introduction, 
    data_preprocessing,
    supervised_learning, 
    unsupervised_learning, 
    neural_networks, 
    nlp_basics, 
    computer_vision, 
    projects,
    generative_ai,
    model_evaluation,
    ai_ethics,
    time_series,
    reinforcement_learning,
    # Advanced Modules
    mlops,
    advanced_nlp,
    recommendation_systems,
    kaggle_guide,
    research_papers,
    # Interactive Tools
    quiz_system,
    progress_dashboard,
    code_playground,
    upload_data,
    # Resources
    video_tutorials,
    cheatsheet,
    interview_prep
)

# Page Configuration
st.set_page_config(
    page_title="AI Masterclass",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Mobile Responsiveness
st.markdown("""
<style>
    /* Global Styling */
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    
    /* Sidebar - Increase touch targets for mobile */
    .stRadio > div {
        gap: 10px;
    }
    .stRadio label {
        background-color: #ffffff;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 5px;
        transition: all 0.2s;
    }
    .stRadio label:hover {
        background-color: #e6f3ff;
        border-color: #2e86de;
        cursor: pointer;
    }

    /* Mobile Text Adjustments */
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; }
        .stMetric { text-align: center; }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("🧠 AI Masterclass")
    st.caption("The Complete AI Learning Platform")
    
    st.markdown("---")
    
    category = st.radio(
        "📂 **Category:**",
        ["📚 Core Modules", "🚀 Advanced Topics", "🔧 Interactive Tools", "📖 Resources"],
        horizontal=False
    )
    
    if category == "📚 Core Modules":
        menu = st.radio(
            "Select Module:",
            [
                "🏠 Home",
                "1. AI Fundamentals",
                "2. Data Preprocessing",
                "3. Supervised Learning",
                "4. Unsupervised Learning",
                "5. Neural Networks",
                "6. Computer Vision",
                "7. NLP Basics",
                "8. Generative AI & LLMs",
                "9. Model Evaluation",
                "10. AI Ethics & Bias",
                "11. Time Series",
                "12. Reinforcement Learning",
                "13. Projects"
            ],
            index=0
        )
    elif category == "🚀 Advanced Topics":
        menu = st.radio(
            "Select Topic:",
            [
                "🚀 MLOps & Deployment",
                "🗣️ Advanced NLP",
                "🎬 Recommendation Systems",
                "🏆 Kaggle Guide",
                "📄 Research Papers"
            ]
        )
    elif category == "🔧 Interactive Tools":
        menu = st.radio(
            "Select Tool:",
            [
                "🎮 Quiz System",
                "📊 Progress Dashboard",
                "🔬 Code Playground",
                "📥 Upload Your Data"
            ]
        )
    else:  # Resources
        menu = st.radio(
            "Select Resource:",
            [
                "📹 Video Tutorials",
                "📋 Cheat Sheet",
                "🎙️ Interview Prep"
            ]
        )
    
    st.markdown("---")
    
    # Progress tracking
    st.markdown("### 📊 Your Progress")
    
    if 'quiz_scores' in st.session_state:
        quizzes_done = len(st.session_state.quiz_scores)
        total_quizzes = 5
        progress_pct = min(quizzes_done / total_quizzes, 1.0)
    else:
        progress_pct = 0
    
    st.progress(progress_pct)
    
    if 'total_xp' in st.session_state:
        st.metric("🏆 XP Earned", st.session_state.total_xp)
    
    st.caption("Complete quizzes to track progress!")
    
    st.markdown("---")
    st.markdown("**📚 18 Modules | 🔧 4 Tools**")

# ======================
# ROUTING
# ======================

# Core Modules
if menu == "🏠 Home":
    st.title("Welcome to the AI Masterclass! 🚀")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌟 The Most Comprehensive AI Learning Platform
        
        This masterclass takes you from **absolute beginner** to **industry-ready professional** with:
        
        #### 📚 18 Learning Modules
        - **Core Foundations** (13 modules): Everything from basics to advanced concepts
        - **Advanced Topics** (5 modules): MLOps, NLP, RecSys, Kaggle, Research Papers
        
        #### 🔧 Interactive Tools
        - 🎮 **Quiz System** - Test knowledge, earn XP, get certificates
        - 🔬 **Code Playground** - Write and run Python in-browser
        - 📥 **Upload Your Data** - Train models on YOUR datasets
        
        #### 📖 Resources
        - 📹 **Video Library** - Curated YouTube tutorials
        - 📋 **Cheat Sheet** - Quick reference for algorithms
        - 🎙️ **Interview Prep** - 50+ ML interview questions
        
        ### 👈 Start with "1. AI Fundamentals" in the sidebar!
        """)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Modules", 18)
        st.metric("Projects", 5)
        st.metric("Quiz Questions", "50+")
        st.metric("Research Papers", 5)
        st.metric("Estimated Time", "~10 hours")
    
    st.markdown("---")
    
    # Track overview
    st.subheader("🎯 Learning Tracks")
    
    tracks = {
        "🔰 Beginner Track": ["AI Fundamentals", "Data Preprocessing", "Supervised Learning", "Model Evaluation", "Projects"],
        "🚀 ML Engineer Track": ["All Core Modules", "MLOps", "Kaggle Guide", "Code Playground"],
        "🧠 NLP Specialist": ["NLP Basics", "Advanced NLP", "Generative AI", "Research Papers (BERT, Transformers)"],
        "🎨 Generative AI Track": ["Neural Networks", "Generative AI", "Research Papers (GANs, Diffusion)"],
    }
    
    cols = st.columns(4)
    for i, (track, modules) in enumerate(tracks.items()):
        with cols[i]:
            st.info(f"**{track}**") # Changed to info box for better mobile visibility
            for m in modules:
                st.caption(f"- {m}")
    
    with st.expander("📋 View All Modules & Durations"):
        import pandas as pd
        data = {
            'Module': ['AI Fundamentals', 'Data Preprocessing', 'Supervised Learning', 'Unsupervised Learning', 
                      'Neural Networks', 'Computer Vision', 'NLP Basics', 'Generative AI', 
                      'Model Evaluation', 'AI Ethics', 'Time Series', 'Reinforcement Learning'],
            'Topics': ['History, ChatGPT', 'Cleaning, Scaling', 'Regression, Trees', 'K-Means, PCA',
                      'Perceptrons, MLP', 'CNNs, Filters', 'BoW, Embeddings', 'LLMs, Prompt Eng',
                      'Metrics, ROC', 'Bias, Fairness', 'Forecasting', 'Q-Learning'],
            'Duration': ['30 min', '20 min', '45 min', '30 min', '40 min', '30 min', 
                        '25 min', '45 min', '30 min', '25 min', '35 min', '40 min']
        }
        df_modules = pd.DataFrame(data)
        st.dataframe(df_modules, use_container_width=True, hide_index=True)

elif menu == "1. AI Fundamentals":
    introduction.show()
elif menu == "2. Data Preprocessing":
    data_preprocessing.show()
elif menu == "3. Supervised Learning":
    supervised_learning.show()
elif menu == "4. Unsupervised Learning":
    unsupervised_learning.show()
elif menu == "5. Neural Networks":
    neural_networks.show()
elif menu == "6. Computer Vision":
    computer_vision.show()
elif menu == "7. NLP Basics":
    nlp_basics.show()
elif menu == "8. Generative AI & LLMs":
    generative_ai.show()
elif menu == "9. Model Evaluation":
    model_evaluation.show()
elif menu == "10. AI Ethics & Bias":
    ai_ethics.show()
elif menu == "11. Time Series":
    time_series.show()
elif menu == "12. Reinforcement Learning":
    reinforcement_learning.show()
elif menu == "13. Projects":
    projects.show()

# Advanced Topics
elif menu == "🚀 MLOps & Deployment":
    mlops.show()
elif menu == "🗣️ Advanced NLP":
    advanced_nlp.show()
elif menu == "🎬 Recommendation Systems":
    recommendation_systems.show()
elif menu == "🏆 Kaggle Guide":
    kaggle_guide.show()
elif menu == "📄 Research Papers":
    research_papers.show()

# Interactive Tools
elif menu == "🎮 Quiz System":
    quiz_system.show()
elif menu == "📊 Progress Dashboard":
    progress_dashboard.show()
elif menu == "🔬 Code Playground":
    code_playground.show()
elif menu == "📥 Upload Your Data":
    upload_data.show()

# Resources
elif menu == "📹 Video Tutorials":
    video_tutorials.show()
elif menu == "📋 Cheat Sheet":
    cheatsheet.show()
elif menu == "🎙️ Interview Prep":
    interview_prep.show()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Created with ❤️ for AI Learners")
with col2:
    st.markdown("**📚 18 Modules | 🔧 4 Tools | 📹 Videos | 🎙️ Interview Prep**")
with col3:
    st.markdown("🧠 AI Masterclass v2.0")
