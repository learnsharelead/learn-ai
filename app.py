import streamlit as st
from streamlit_option_menu import option_menu
from modules import (
    introduction, data_preprocessing, supervised_learning, unsupervised_learning, 
    neural_networks, nlp_basics, computer_vision, projects, generative_ai,
    model_evaluation, ai_ethics, time_series, reinforcement_learning,
    mlops, advanced_nlp, recommendation_systems, kaggle_guide, research_papers,
    quiz_system, progress_dashboard, code_playground, upload_data,
    video_tutorials, cheatsheet, interview_prep
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Masterclass",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# ENTERPRISE-GRADE CSS SYSTEM
# =============================================================================
st.markdown("""
<style>
    /* ==========================================================================
       RESET & BASICS
       ========================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Remove Streamlit Bloat */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    .stDeployButton {display: none;}
    
    /* App Background */
    .stApp {
        background-color: #fbfbfd; /* Apple System Gray 6 */
    }
    
    /* ==========================================================================
       DENSE LAYOUT & WHITESPACE REMOVAL
       ========================================================================== */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important; /* WIDE LAYOUT */
        margin: 0 auto !important;
    }
    
    header { visibility: hidden !important; }

    /* ==========================================================================
       CUSTOM NAVBAR COMPONENT
       ========================================================================== */
    /* We will use Streamlit's container for this, styled to float */
    div[data-testid="stVerticalBlock"] > div:first-child {
        /* This targets the top area */
        z-index: 999;
    }

    /* ==========================================================================
       DENSE LAYOUT & WHITESPACE REMOVAL
       ========================================================================== */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    header { visibility: hidden !important; }

    /* ==========================================================================
       TYPOGRAPHY - Dense & Modern (Refined)
       ========================================================================== */
    h1 {
        font-family: -apple-system, sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important; /* Reduced from 1.6rem */
        letter-spacing: -0.01em !important;
        color: #1d1d1f !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0 !important;
    }
    
    h2, h3 {
        color: #1d1d1f !important;
        margin-top: 0.6rem !important; /* Reduced margin */
        margin-bottom: 0.3rem !important;
    }

    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    
    p, li, label {
        font-size: 13.5px !important;
        line-height: 1.4 !important;
        color: #3b3b3b !important;
    }
    
    /* Exclude Material Icons from being touched by global span rules if possible,
       but better to be specific. Restricting strict styling to markdown paragraphs. */
    .stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol, .stMarkdown label {
        font-size: 13.5px !important;
        line-height: 1.4 !important;
        color: #3b3b3b !important;
    }
    
    /* Input fields Text */
    input, textarea {
        font-size: 13.5px !important;
    }
    
    /* Specific overrides for expanders to ensure icons work */
    .streamlit-expanderHeader {
        font-size: 14px !important;
        color: #1d1d1f !important;
    }
    
    /* Remove generic span/div overrides that break icons */
    :not(.material-icons) > span {
         /* Safety check: do nothing globally */
    }

    /* ==========================================================================
       APPLE BACKGROUND SYSTEM
       ========================================================================== */
    /* Main Page Background - Apple 'Smoke' */
    .stApp {
        background-color: #f5f5f7 !important;
    }
    
    /* Ensure text is standard dark gray */
    .stMarkdown, p, h1, h2, h3, li, label, div {
        color: #1d1d1f !important;
    }

    /* ==========================================================================
       "VIBRANT PILL" TABS (Lighter Apple Blue)
       ========================================================================== */
    .stTabs {
        margin-top: 0px;
    }
    
    /* The Track */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.6); /* Translucent white track */
        backdrop-filter: blur(10px);
        padding: 4px;
        border-radius: 12px;
        border-bottom: none !important;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: flex-start;
        flex-wrap: wrap;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Subtle depth */
    }

    /* The Tab Item */
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        padding: 0 16px;
        border-radius: 8px;
        border: none !important;
        background-color: transparent !important;
        color: #6e6e73;
        font-weight: 500;
        font-size: 13px;
        flex-grow: 1;
        max-width: 180px;
    }

    /* ==========================================================================
       HIERARCHICAL TAB COLOR SYSTEM
       ========================================================================== */
    
    /* LEVEL 1: MAIN TABS (Blue) */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        color: #0d47a1 !important;
        box-shadow: 0 2px 6px rgba(13, 71, 161, 0.15);
        border: 1px solid rgba(13, 71, 161, 0.1) !important;
    }

    /* LEVEL 2: NESTED TABS (e.g., Curriculum Modules - Soft Orange) */
    div[data-baseweb="tab-panel"] .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
        color: #e65100 !important;
        box-shadow: 0 2px 6px rgba(230, 81, 0, 0.15);
        border: 1px solid rgba(230, 81, 0, 0.1) !important;
    }

    /* LEVEL 3: DEEP NESTED TABS (e.g., Content Topics - Soft Teal) */
    div[data-baseweb="tab-panel"] div[data-baseweb="tab-panel"] .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%) !important;
        color: #004d40 !important;
        box-shadow: 0 2px 6px rgba(0, 77, 64, 0.15);
        border: 1px solid rgba(0, 77, 64, 0.1) !important;
    }
    
    /* ==========================================================================
       CARDS & SURFACES (White on Gray)
       ========================================================================== */
    .feature-box, .content-card {
        background: #ffffff !important; /* Pure white cards */
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 2px 10px rgba(0,0,0,0.03); /* Soft shadow */
    }
    .bento-box:hover {
        transform: scale(1.02);
    }
    .bento-blue { background: linear-gradient(135deg, #e0f2ff 0%, #dbeafe 100%); }
    .bento-purple { background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); }
    .bento-green { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); }
    .bento-orange { background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); }
    
    .bento-icon { font-size: 24px; margin-bottom: 10px; }
    .bento-title { font-weight: 700; font-size: 16px; color: #1e3a8a; margin-bottom: 4px; }
    .bento-desc { font-size: 13px; color: #475569; line-height: 1.3; }

    hr { margin: 1rem 0 !important; border-top: 1px solid #e5e5ea; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    button { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
c_head1, c_head2 = st.columns([0.1, 0.9])
with c_head1:
    st.markdown("<div style='font-size:24px;'></div>", unsafe_allow_html=True)
with c_head2:
    st.markdown("<h3 style='margin: 0; padding-top: 5px; font-weight: 700;'>AI Masterclass</h3>", unsafe_allow_html=True)

# =============================================================================
# UNIFIED NAVIGATION (Vibrant Pills)
# =============================================================================

# Top-level Tabs
main_tabs = st.tabs(["🏠 Home", "📚 Curriculum", "🛠️ Lab", "📑 Reference"])

# --- TAB 1: HOME ---
with main_tabs[0]:
    # Ultra Compact Hero
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 2.5rem !important;'>Master AI.<br>Design the Future.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666;'>Interactive learning. Zero fluff.</p>", unsafe_allow_html=True)
        st.button("Start Now ⚡", key="hero_start")
    
    st.markdown("---")
    
    # Colorful Bento Grid
    st.markdown("### Your Toolkit")
    cols = st.columns(4)
    with cols[0]:
        st.markdown("""
        <div class="bento-box bento-blue">
            <div class="bento-icon">🤖</div>
            <div class="bento-title">Fundamentals</div>
            <div class="bento-desc">Zero to Hero. Logic & Math.</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="bento-box bento-purple">
            <div class="bento-icon">🧠</div>
            <div class="bento-title">Neural Nets</div>
            <div class="bento-desc">Build your own Brain.</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="bento-box bento-orange">
            <div class="bento-icon">🎨</div>
            <div class="bento-title">Generative AI</div>
            <div class="bento-desc">LLMs & Diffusion.</div>
        </div>""", unsafe_allow_html=True)
    with cols[3]:
        st.markdown("""
        <div class="bento-box bento-green">
            <div class="bento-icon">📊</div>
            <div class="bento-title">Data Lab</div>
            <div class="bento-desc">Real-world projects.</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


# --- TAB 2: CURRICULUM ---
with main_tabs[1]:
    # Nested Tabs for Modules
    mod_tabs = st.tabs([
        "Fundamentals", "Data", "Supervised", "Unsupervised", 
        "Neural Nets", "NLP", "Generative AI", "Ethics"
    ])
    
    with mod_tabs[0]: introduction.show()
    with mod_tabs[1]: data_preprocessing.show()
    with mod_tabs[2]: supervised_learning.show()
    with mod_tabs[3]: unsupervised_learning.show()
    with mod_tabs[4]: neural_networks.show()
    with mod_tabs[5]: nlp_basics.show()
    with mod_tabs[6]: generative_ai.show()
    with mod_tabs[7]: ai_ethics.show()


# --- TAB 3: LAB ---
with main_tabs[2]:
    lab_tabs = st.tabs(["Playground", "Quiz", "Upload", "Projects"])
    with lab_tabs[0]: code_playground.show()
    with lab_tabs[1]: quiz_system.show()
    with lab_tabs[2]: upload_data.show()
    with lab_tabs[3]: projects.show()


# --- TAB 4: REFERENCE ---
with main_tabs[3]:
    ref_tabs = st.tabs(["CheatSheet", "Videos", "Interviews", "Papers", "MLOps"])
    with ref_tabs[0]: cheatsheet.show()
    with ref_tabs[1]: video_tutorials.show()
    with ref_tabs[2]: interview_prep.show()
    with ref_tabs[3]: research_papers.show()
    with ref_tabs[4]: mlops.show()

# =============================================================================
# GLOBAL FOOTER
# =============================================================================
st.markdown("""
<div style="text-align: center; margin-top: 5rem; padding-top: 2rem; border-top: 1px solid #e5e5ea; color: #86868b; font-size: 12px;">
    Copyright © 2025 AI Masterclass Inc. All rights reserved. <br>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Privacy Policy</a>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Terms of Use</a>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Site Map</a>
</div>
""", unsafe_allow_html=True)
