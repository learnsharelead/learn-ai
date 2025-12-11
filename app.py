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
# HEADER (New Brand Identity)
# =============================================================================
c_head1, c_head2 = st.columns([0.15, 0.85])
with c_head1:
    st.markdown("<div style='font-size:28px; padding-top:2px;'>🧬</div>", unsafe_allow_html=True)
with c_head2:
    st.markdown("<h3 style='margin: 0; padding-top: 5px; font-weight: 800; letter-spacing: -0.5px;'>NEXUS <span style='color:#6e6e73; font-weight:400;'>AI Academy</span></h3>", unsafe_allow_html=True)

# =============================================================================
# UNIFIED NAVIGATION (Vibrant Pills)
# =============================================================================

# Top-level Tabs
main_tabs = st.tabs(["🏠 Home", "📚 Curriculum", "🛠️ Lab", "📑 Reference"])

# --- TAB 1: HOME (Rebranded & Sales Focused) ---
with main_tabs[0]:
    # --- HERO SECTION (Centered & Unified) ---
    st.markdown("""
    <style>
        .hero-container {
            text-align: center;
            padding: 3rem 0 2rem 0;
            margin: 0 auto;
            max-width: 900px;
        }
        .hero-text {
            background: linear-gradient(90deg, #1d1d1f 0%, #2563eb 50%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.8rem !important;
            font-weight: 800;
            letter-spacing: -2px;
            line-height: 1.1;
            margin-bottom: 0.8rem;
            white-space: nowrap; /* Force single line */
        }
        .sub-hero {
            font-size: 1.4rem !important;
            color: #6e6e73;
            font-weight: 500;
            letter-spacing: -0.5px;
        }
    </style>
    <div class="hero-container">
        <div class="hero-text">Build the Brain. Design the Future.</div>
        <div class="sub-hero">Zero Paywalls. 100% Open Source. The New Standard for AI Mastery.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered CTA
    _, c_cta, _ = st.columns([1, 1, 1])
    with c_cta:
        st.button("🚀 Start Your Journey", key="hero_cta_main", use_container_width=True)
    
    st.markdown("---")
    
    # --- VIBRANT STYLING (Home Page Exclusive) ---
    st.markdown("""
    <style>
        .vibrant-card {
            padding: 1.5rem;
            border-radius: 20px;
            color: white;
            height: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
        }
        .vibrant-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        .card-blue { background: linear-gradient(135deg, #2563eb 0%, #00C6FF 100%); }
        .card-purple { background: linear-gradient(135deg, #7c3aed 0%, #f43f5e 100%); }
        .card-orange { background: linear-gradient(135deg, #f59e0b 0%, #ff6b6b 100%); }
        .card-green { background: linear-gradient(135deg, #10b981 0%, #34d399 100%); }
        
        .bento-icon-lg { font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.9; }
        .bento-title-lg { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.2rem; }
        .bento-desc-lg { font-size: 0.9rem; opacity: 0.9; font-weight: 500; }
        
        .news-card-pro {
            background: white;
            border-radius: 16px;
            padding: 0;
            height: 100%;
            min-height: 200px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s;
            border: 1px solid rgba(0,0,0,0.03);
            overflow: hidden;
        }
        .news-card-pro:hover { transform: translateY(-3px); }
        .news-header {
            padding: 12px 16px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: white;
        }
        .news-body { padding: 16px; }
        .news-date { color: #888; font-size: 11px; margin-bottom: 8px; display: block; }
    </style>
    """, unsafe_allow_html=True)
    
    # --- FEATURE BENTO GRID (Vibrant) ---
    st.markdown("<h3 style='margin-bottom:1.5rem; font-weight:700;'>🚀 The Nexus Toolkit</h3>", unsafe_allow_html=True)
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown("""
        <div class="vibrant-card card-blue">
            <div class="bento-icon-lg">🦾</div>
            <div class="bento-title-lg">Core AI</div>
            <div class="bento-desc-lg">Math, Logic & Python.</div>
        </div>""", unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown("""
        <div class="vibrant-card card-purple">
            <div class="bento-icon-lg">🧠</div>
            <div class="bento-title-lg">Deep Learning</div>
            <div class="bento-desc-lg">CNNs & Transformers.</div>
        </div>""", unsafe_allow_html=True)
        
    with cols[2]:
        st.markdown("""
        <div class="vibrant-card card-orange">
            <div class="bento-icon-lg">🎨</div>
            <div class="bento-title-lg">Generative</div>
            <div class="bento-desc-lg">Prompts & Diffusion.</div>
        </div>""", unsafe_allow_html=True)
        
    with cols[3]:
        st.markdown("""
        <div class="vibrant-card card-green">
            <div class="bento-icon-lg">⚡</div>
            <div class="bento-title-lg">Ops & Scale</div>
            <div class="bento-desc-lg">Deploy & Ethics.</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- LATEST NEWS SECTION (Pro Cards) ---
    st.markdown("<h3 style='margin-bottom:1.5rem; font-weight:700;'>📰 Live Intelligence Feed</h3>", unsafe_allow_html=True)
    
    news_items = [
        {
            "title": "Gemini 2.0: Multimodal King?",
            "date": "DEC 10 • MODELS",
            "grad": "linear-gradient(90deg, #4f46e5, #00C6FF)",
            "text": "Google's new model crushes benchmarks. Is this the end of the text-only era?"
        },
        {
            "title": "NVIDIA B200 Chips Arrive",
            "date": "DEC 08 • HARDWARE",
            "grad": "linear-gradient(90deg, #f59e0b, #ff6b6b)",
            "text": "The 30x speed boost is real. Data centers are racing to upgrade for Trillion-param models."
        },
        {
            "title": "New Global AI Safety Treaty",
            "date": "DEC 05 • POLICY",
            "grad": "linear-gradient(90deg, #10b981, #059669)",
            "text": "15 Nations sign accord to ban autonomous weapons and limit generative deepfakes."
        }
    ]
    
    n_cols = st.columns(3)
    for i, news in enumerate(news_items):
        with n_cols[i]:
            st.markdown(f"""
            <div class="news-card-pro">
                <div class="news-header" style="background: {news['grad']};">
                    {news['date']}
                </div>
                <div class="news-body">
                    <h4 style="margin: 0 0 8px 0; font-size: 17px; font-weight: 700; color: #111;">{news['title']}</h4>
                    <p style="font-size: 14px; color: #555; line-height: 1.5; margin: 0;">{news['text']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- MISSION & CONTACT SECTION ---
    c_mission, c_contact = st.columns([2, 1])
    
    with c_mission:
        st.markdown("""
        <div class="content-card">
            <h3 style="margin-top:0;">🌐 Open Learning Initiative</h3>
            <p><strong>Education should be free and accessible to all.</strong></p>
            <p style="font-size:14px; color:#444;">
                Nexus AI is dedicated to democratizing Artificial Intelligence education. 
                No paywalls, no hidden fees—just pure, high-quality knowledge for the curious mind.
                Whether you're a student, researcher, or hobbyist, this platform is your open playground.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with c_contact:
        st.markdown("""
        <div class="content-card" style="background:#1d1d1f !important; color:white !important;">
            <h3 style="margin-top:0; color:white !important;">📬 Connect</h3>
            <p style="color:#a1a1a6 !important;">Feedback? Questions? Ideas?</p>
            <div style="margin-top:15px; font-size:13px; color:#ffffff;">
                <b>Vikas Singh</b><br>
                Creator & Educator<br>
                <br>
                ✉️ <a href="mailto:vikas.singh.info@gmail.com" style="color:#2997ff; text-decoration:none;">vikas.singh.info@gmail.com</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- FOOTER ---
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #86868b; font-size: 12px;">
        <hr>
        &copy; 2025 Vikas Singh • Made with ❤️ for the AI Community<br>
        Designed with 🧬 Nexus Design System v2.0
    </div>
    """, unsafe_allow_html=True)


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
