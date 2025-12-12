import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables (must be before other imports)
load_dotenv()

import streamlit.components.v1 as components
import json
from streamlit_option_menu import option_menu
from utils.news_fetcher import fetch_ai_news
from modules import (
    introduction, data_preprocessing, supervised_learning, unsupervised_learning, 
    neural_networks, nlp_basics, computer_vision, projects, generative_ai,
    model_evaluation, ai_ethics, time_series, reinforcement_learning,
    mlops, advanced_nlp, recommendation_systems, kaggle_guide, research_papers,
    quiz_system, progress_dashboard, code_playground, upload_data,
    video_tutorials, cheatsheet, interview_prep, model_arena, nexus_tutor, prompt_lab,
    neural_viz_3d,
    # Developer Track
    ai_dev_stack, langchain_langraph, rag_tutorial, agentic_ai, mcp_tutorial, ai_testing,
    fine_tuning, ai_security, multimodal_ai, red_teaming, bias_fairness,
    cost_optimization, observability, openai_assistants, tools_deep_dive,
    multi_agent_systems, synthetic_data, performance_testing, ab_testing
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
    page_title="Synapse AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GOOGLE ANALYTICS 4 TRACKING
# =============================================================================
import os

GA_MEASUREMENT_ID = os.getenv("GA_MEASUREMENT_ID", "")

if GA_MEASUREMENT_ID:
    ga_code = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{GA_MEASUREMENT_ID}');
    </script>
    """
    components.html(ga_code, height=0)

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
    .bento-orange { background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); }
    .bento-green { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); }
    
    /* ==========================================================================
       MOBILE RESPONSIVENESS
       ========================================================================== */
    
    /* Tablets and below (768px) */
    @media (max-width: 768px) {
        /* Hero Section - Reduce font sizes */
        h1[style*="5.5rem"] {
            font-size: 3rem !important;
        }
        
        p[style*="1.6rem"] {
            font-size: 1.2rem !important;
        }
        
        /* Reduce padding on hero */
        div[style*="padding: 5rem"] {
            padding: 2rem 1rem !important;
        }
        
        /* Make columns stack on mobile */
        .row-widget.stHorizontalBlock {
            flex-direction: column !important;
        }
        
        .row-widget.stHorizontalBlock > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* Adjust max width for mobile */
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* Mobile phones (480px and below) */
    @media (max-width: 480px) {
        /* Hero - Even smaller */
        h1[style*="5.5rem"] {
            font-size: 2.2rem !important;
        }
        
        p[style*="1.6rem"] {
            font-size: 1rem !important;
        }
        
        /* Compact padding */
        div[style*="padding: 5rem"] {
            padding: 1.5rem 0.5rem !important;
        }
        
        /* Tabs wrap better */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            max-width: none !important;
            flex-grow: 1;
            min-width: 100px;
        }
    }
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
# =============================================================================
# HEADER (New Brand Identity)
# =============================================================================
st.markdown("""
<div style="
    display: flex; 
    align-items: center; 
    justify-content: center; 
    gap: 15px; 
    padding-bottom: 15px;
">
    <div style="font-size: 40px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">🧬</div>
    <h1 style='
        margin: 0; 
        padding: 0; 
        font-size: 38px; 
        font-weight: 800; 
        letter-spacing: -1px; 
        color: #111827;
        line-height: 1;
    '>
        Veda <span style='background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AI</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# UNIFIED NAVIGATION (Vibrant Pills)
# =============================================================================

# =============================================================================
# UNIFIED NAVIGATION (Controllable State)
# =============================================================================

# Define Navigation Options (Text Only, Icons handled by option_menu)
nav_options = ["Home", "Curriculum", "Developers", "Lab", "Reference", "Dashboard"]
icons = ["house-fill", "journal-code", "code-slash", "cpu-fill", "archive-fill", "grid-1x2-fill"]

# Navigation State Management
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = nav_options[0]

def navigate_to(page_name):
    st.session_state.nav_selection = page_name

# Render Option Menu
selected_nav = option_menu(
    menu_title=None,
    options=nav_options,
    icons=icons,
    default_index=nav_options.index(st.session_state.nav_selection),
    orientation="horizontal",
    styles={
        "container": {
            "padding": "8px", 
            "background-color": "#ffffff", 
            "border-radius": "16px",
            "box-shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)",
            "border": "1px solid rgba(0,0,0,0.05)"
        },
        "icon": {"font-size": "18px"}, # Color removed to allow inheritance
        "nav-link": {
            "font-size": "16px", 
            "text-align": "center", 
            "margin": "0px 4px", 
            "--hover-color": "#eff6ff",
            "border-radius": "10px",
            "padding": "10px 16px",
            "font-weight": "500",
            "color": "#64748b" # Default color set here
        },
        "nav-link-selected": {
            "background-color": "#2563eb", 
            "color": "white", 
            "font-weight": "600",
            "box-shadow": "0 4px 12px rgba(37, 99, 235, 0.3)"
        },
    },
    key='navigation_menu',
    on_change=lambda key: navigate_to(st.session_state[key]) # Sync state
)



# --- SIDEBAR TUTOR ---
nexus_tutor.show()

# --- HOME TAB ---
if st.session_state.nav_selection == "Home":
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. SYSTEM CAPABILITIES (ULTRA-COMPACT BENTO ROW)
    st.markdown("### 🌌 The Neural Grid")
    
    # Ultra-Compact Layout: 4 Columns in 1 Row
    modules_map = [
        {"icon": "🎓", "title": "Academy", "desc": "Theory & Deep Learning", "bg": "#eff6ff", "border": "#bfdbfe", "shadow": "rgba(59, 130, 246, 0.1)", "target": "Curriculum"},
        {"icon": "💻", "title": "Dev Track", "desc": "RAG, Agents & MCP", "bg": "#f5f3ff", "border": "#ddd6fe", "shadow": "rgba(139, 92, 246, 0.1)", "target": "Developers"},
        {"icon": "🛠️", "title": "Neural Lab", "desc": "Arena, 3D & Playground", "bg": "#fff7ed", "border": "#fed7aa", "shadow": "rgba(249, 115, 22, 0.1)", "target": "Lab"},
        {"icon": "🧠", "title": "Research", "desc": "Papers & Reference", "bg": "#f0fdf4", "border": "#bbf7d0", "shadow": "rgba(16, 185, 129, 0.1)", "target": "Reference"}
    ]
    
    # Responsive Grid: 4 columns on desktop, automatically stacks on mobile
    cols = st.columns(4, gap="medium")
    
    for i, mod in enumerate(modules_map):
        col_index = i % 4  # Distribute across 4 columns
        with cols[col_index]:
            # Static Card (No cursor pointer)
            st.markdown(f"""
            <div style="background: {mod['bg']}; border: 1px solid {mod['border']}; padding: 15px; border-radius: 12px; min-height: 130px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; box-shadow: 0 2px 5px {mod['shadow']};">
                <div style="font-size: 2rem; margin-bottom: 5px;">{mod['icon']}</div>
                <div style="font-weight: 700; color: #111; font-size: 1rem; margin-bottom: 2px;">{mod['title']}</div>
                <div style="font-size: 0.8rem; color: #666; line-height: 1.2;">{mod['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button(f"Open", key=f"nav_btn_{i}", use_container_width=True, on_click=lambda t=mod['target']: navigate_to(t))

    st.markdown("---")

    # 2. VISUAL INTELLIGENCE FEED
    st.markdown("### 📡 Global Intelligence Feed")
    with st.spinner("Analyzing global signals..."):
        latest_news = fetch_ai_news(limit=15)
    
    news_json = json.dumps(latest_news)
    
    # ... (Keep Carousel HTML logic same, omitted for brevity, just ensure it renders) ...
    # Re-inserting Carousel HTML for completeness since replace overwrites block
    carousel_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            body {{ font-family: 'Inter', sans-serif; margin: 0; overflow: hidden; background: transparent; }}
            .carousel {{ position: relative; width: 100%; height: 380px; perspective: 1000px; }}
            .carousel-track-container {{ overflow: hidden; width: 100%; height: 100%; padding: 10px 0; }}
            .carousel-track {{ display: flex; transition: transform 0.6s cubic-bezier(0.25, 1, 0.5, 1); gap: 20px; padding-left: 10px; }}
            .card {{ flex: 0 0 300px; background: rgba(255, 255, 255, 0.9); border-radius: 20px; overflow: hidden; text-decoration: none; color: inherit; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid rgba(255,255,255,0.6); transition: transform 0.3s ease; height: 340px; display: flex; flex-direction: column; }}
            .card:hover {{ transform: translateY(-5px); box-shadow: 0 12px 25px rgba(0,0,0,0.1); }}
            .card-image {{ height: 160px; width: 100%; background-size: cover; background-position: center; position: relative; }}
            .badge {{ position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.6); color: white; padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase; backdrop-filter: blur(4px); }}
            .card-content {{ padding: 16px; display: flex; flex-direction: column; flex-grow: 1; justify-content: space-between; }}
            .date {{ font-size: 11px; color: #888; margin-bottom: 6px; font-weight: 600; }}
            .title {{ font-size: 16px; font-weight: 700; color: #111; line-height: 1.4; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; display: -webkit-box; }}
            .read-more {{ margin-top: 12px; font-size: 13px; color: #2563eb; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="carousel" id="carousel">
            <div class="carousel-track-container">
                <div class="carousel-track" id="track"></div>
            </div>
        </div>
        <script>
            const newsData = {news_json};
            const track = document.getElementById('track');
            newsData.forEach(item => {{
                const card = document.createElement('a');
                card.className = 'card'; card.href = item.link; card.target = '_blank';
                card.innerHTML = `<div class="card-image" style="background-image: url('${{item.image}}');"><div class="badge">${{item.source}}</div></div><div class="card-content"><div><div class="date">${{item.date}}</div><div class="title">${{item.title}}</div></div><div class="read-more">Read Article →</div></div>`;
                track.appendChild(card);
            }});
            let idx = 0; const w = 320;
            setInterval(() => {{
                idx++; if (idx > newsData.length - 2) idx = 0;
                track.style.transform = `translateX(-${{idx * w}}px)`;
            }}, 4000);
        </script>
    </body>
    </html>
    """
    components.html(carousel_html, height=400, scrolling=False)

# --- TAB 2: CURRICULUM ---
if st.session_state.nav_selection == "Curriculum":
    # Nested Tabs for Modules
    mod_tabs = st.tabs([
        "Fundamentals", "Data", "Supervised", "Unsupervised", "Neural Nets", 
        "Computer Vision", "NLP", "Advanced NLP", "Time Series", "Rec Sys", "Reinforcement", "Generative AI", "Ethics"
    ])
    
    with mod_tabs[0]: introduction.show()
    with mod_tabs[1]: data_preprocessing.show()
    with mod_tabs[2]: supervised_learning.show()
    with mod_tabs[3]: unsupervised_learning.show()
    with mod_tabs[4]: neural_networks.show()
    with mod_tabs[5]: computer_vision.show()
    with mod_tabs[6]: nlp_basics.show()
    with mod_tabs[7]: advanced_nlp.show()
    with mod_tabs[8]: time_series.show()
    with mod_tabs[9]: recommendation_systems.show()
    with mod_tabs[10]: reinforcement_learning.show()
    with mod_tabs[11]: generative_ai.show()
    with mod_tabs[12]: ai_ethics.show()


# --- TAB 3: DEVELOPER TRACK ---
if st.session_state.nav_selection == "Developers":
    st.markdown("""
    ### 💻 Developer & Tester Track
    Build production-grade AI applications. From API calls to Agentic systems.
    """)
    dev_tabs = st.tabs([
        "🛠️ AI Stack", "🔗 LangChain", "📚 RAG", "🤖 Agents", "🔌 MCP", "🧪 Testing",
        "🎛️ Fine-Tuning", "🔐 Security", "👁️ Multi-Modal", "🔴 Red Teaming", "⚖️ Fairness",
        "💰 Cost Opt", "📊 Observability", "🤖 Assistants API", "🛠️ Deep Dives",
        "👥 Multi-Agent", "🎲 Synthetic Data", "⚡ Perf Test", "📈 A/B Test"
    ])
    with dev_tabs[0]: ai_dev_stack.show()
    with dev_tabs[1]: langchain_langraph.show()
    with dev_tabs[2]: rag_tutorial.show()
    with dev_tabs[3]: agentic_ai.show()
    with dev_tabs[4]: mcp_tutorial.show()
    with dev_tabs[5]: ai_testing.show()
    with dev_tabs[6]: fine_tuning.show()
    with dev_tabs[7]: ai_security.show()
    with dev_tabs[8]: multimodal_ai.show()
    with dev_tabs[9]: red_teaming.show()
    with dev_tabs[10]: bias_fairness.show()
    with dev_tabs[11]: cost_optimization.show()
    with dev_tabs[12]: observability.show()
    with dev_tabs[13]: openai_assistants.show()
    with dev_tabs[14]: tools_deep_dive.show()
    with dev_tabs[15]: multi_agent_systems.show()
    with dev_tabs[16]: synthetic_data.show()
    with dev_tabs[17]: performance_testing.show()
    with dev_tabs[18]: ab_testing.show()


# --- TAB 4: LAB ---
if st.session_state.nav_selection == "Lab":
    lab_tabs = st.tabs(["⚔️ Arena", "🧊 3D Net", "🧪 Prompt Lab", "Playground", "Quiz", "Upload", "Projects"])
    with lab_tabs[0]: model_arena.show()
    with lab_tabs[1]: neural_viz_3d.show()
    with lab_tabs[2]: prompt_lab.show()
    with lab_tabs[3]: code_playground.show()
    with lab_tabs[4]: quiz_system.show()
    with lab_tabs[5]: upload_data.show()
    with lab_tabs[6]: projects.show()


# --- TAB 5: REFERENCE ---
if st.session_state.nav_selection == "Reference":
    ref_tabs = st.tabs(["CheatSheet", "Videos", "Interviews", "Papers", "MLOps"])
    with ref_tabs[0]: cheatsheet.show()
    with ref_tabs[1]: video_tutorials.show()
    with ref_tabs[2]: interview_prep.show()
    with ref_tabs[3]: research_papers.show()
    with ref_tabs[4]: mlops.show()

# --- TAB 6: DASHBOARD ---
if st.session_state.nav_selection == "Dashboard":
    progress_dashboard.show()

# =============================================================================
# GLOBAL FOOTER
# =============================================================================
st.markdown("""
<div style="text-align: center; margin-top: 5rem; padding-top: 2rem; border-top: 1px solid #e5e5ea; color: #86868b; font-size: 12px;">
    Copyright © 2025 vikas.singh.info@gmail.com. All rights reserved. <br>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Privacy Policy</a>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Terms of Use</a>
    <a href="#" style="color: #86868b; text-decoration: none; margin: 0 10px;">Site Map</a>
</div>
""", unsafe_allow_html=True)
