import streamlit as st

def apply_apple_design():
    """
    Applies an Apple-inspired Design System to the Streamlit app.
    Focus: Clean typography, 'Pill' buttons, soft shadows, high contrast text.
    Based on docs/design_system.md specifications.
    """
    
    st.markdown("""
        <style>
        /* --------------------------------------------------------------------------------
           TYPOGRAPHY SYSTEM (San Francisco Alternate: Inter)
           -------------------------------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
            color: #1d1d1f; /* Apple Primary Text */
            background-color: #f5f5f7; /* Apple Off-White Background */
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1d1d1f;
            letter-spacing: -0.01em; /* Tight tracking for headlines */
        }

        h1 {
            font-weight: 700;
            font-size: 3rem !important;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-weight: 600;
            font-size: 2rem !important;
            margin-top: 1.5rem;
        }

        h3 {
            font-weight: 600;
            font-size: 1.4rem !important;
        }
        
        /* --------------------------------------------------------------------------------
           LAYOUT & BACKGROUND
           -------------------------------------------------------------------------------- */
        .block-container {
            max-width: 980px; /* Apple Content Width */
            padding-top: 3rem;
            padding-bottom: 5rem;
        }

        .stApp {
            background-color: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background-color: #fbfbfd; /* Very subtle sidebar */
            border-right: 1px solid #d2d2d7;
        }
        
        /* --------------------------------------------------------------------------------
           COMPONENTS: BUTTONS (The "Pill" Shape)
           -------------------------------------------------------------------------------- */
        .stButton > button {
            background-color: #0071e3; /* Apple Blue */
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 500;
            border-radius: 980px !important; /* Pill shape */
            border: none;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #0077ED; /* Slightly lighter on hover */
            color: #FFFFFF;
            transform: scale(1.02); /* Micro-interaction scale */
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Soft shadow */
        }
        
        .stButton > button:active {
            transform: scale(0.98);
        }

        /* Secondary Buttons (Outlined) - Approximation via Streamlit */
        /* Note: Streamlit doesn't distinguish button types easily via CSS alone, 
           so we treat all primary buttons as Call-to-Actions */

        /* --------------------------------------------------------------------------------
           COMPONENTS: CARDS & INFO BOXES
           -------------------------------------------------------------------------------- */
        
        /* Info/Success/Warning Boxes -> Apple Cards */
        .stAlert {
            background-color: #ffffff;
            border: 1px solid #d2d2d7;
            border-radius: 12px; /* Apple Radius */
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            color: #1d1d1f;
        }
        
        /* Expanders -> Clean Lists */
        .stExpander {
            border: none;
            border-bottom: 1px solid #d2d2d7;
            border-radius: 0px;
        }
        
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1d1d1f;
        }

        /* --------------------------------------------------------------------------------
           NAVIGATION menus (Radio Buttons)
           -------------------------------------------------------------------------------- */
        .stRadio > div {
            gap: 8px;
        }
        
        .stRadio label {
            padding: 8px 12px;
            border-radius: 8px;
            transition: background 0.2s;
            cursor: pointer;
            font-size: 15px;
            color: #1d1d1f;
        }
        
        .stRadio label:hover {
            background-color: #f5f5f7; /* Hover state */
        }

        /* Selected Tab/Menu Item */
        .stRadio div[aria-checked="true"] + div {
             color: #0071e3; /* Active Blue */
             font-weight: 600;
        }

        /* --------------------------------------------------------------------------------
           DATAFRAMES & TABLES
           -------------------------------------------------------------------------------- */
        [data-testid="stDataFrame"] {
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        
        /* --------------------------------------------------------------------------------
           UTILITIES
           -------------------------------------------------------------------------------- */
        hr {
            margin: 2.5rem 0;
            border: none;
            border-top: 1px solid #d2d2d7; /* Subtle divider */
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
            font-weight: 700;
            font-size: 2.5rem;
            color: #1d1d1f;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #86868b; /* Apple Gray */
        }

        </style>
    """, unsafe_allow_html=True)
