import streamlit as st
import json

def setup_seo_routing(nav_options, default_selection):
    """
    Syncs the navigation selection with URL query parameters for SEO-friendly linking.
    Returns the effective selection (from URL or default).
    """
    # 1. Get query params
    query_params = st.query_params
    url_nav = query_params.get("nav", None)
    
    # 2. Determine initial selection
    if url_nav in nav_options:
        if "nav_selection" not in st.session_state:
            st.session_state.nav_selection = url_nav
        return url_nav
    else:
        # Default or fallback
        if "nav_selection" not in st.session_state:
            st.session_state.nav_selection = default_selection
        return st.session_state.nav_selection

def update_url(selection):
    """
    Updates the URL query parameter to match the current selection.
    """
    st.query_params["nav"] = selection

def inject_seo_meta(title, description, keywords=None, schema_type="TechArticle"):
    """
    Injects JSON-LD Structured Data for Google Rich Results.
    This helps search engines understand the context of the dynamic page.
    """
    schema = {
        "@context": "https://schema.org",
        "@type": schema_type,
        "headline": title,
        "description": description,
        "author": {
            "@type": "Organization",
            "name": "Veda AI"
        },
        "publisher": {
            "@type": "Organization",
            "name": "Veda AI",
            "logo": {
                "@type": "ImageObject",
                "url": "https://via.placeholder.com/150" # Replace with actual logo URL if available
            }
        }
    }
    
    if keywords:
        schema["keywords"] = keywords

    # Minimal invisible HTML injection
    st.markdown(f"""
    <script type="application/ld+json">
    {json.dumps(schema)}
    </script>
    <div style="display:none;">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
