import streamlit as st

# Helper for resilient video rendering
def render_video(url, caption):
    st.subheader(caption.split(" - ")[0] if " - " in caption else caption)
    try:
        st.video(url)
    except:
        st.error("Video error")
    
    st.caption(caption)
    st.link_button("📺 Watch on YouTube", url)

def show():
    st.title("📹 Video Tutorials")
    
    st.markdown("""
    Learn from curated video content! Each video is selected to complement the written tutorials.
    **Note:** If a video says "Unavailable", click the **Watch on YouTube** button.
    """)
    
    # Video categories updated with new tracks
    tabs = st.tabs(["🎓 Fundamentals", "🔥 Machine Learning", "🧠 Deep Learning", "🤖 Generative AI", "💻 Developers", "🧪 Testers", "💼 Career"])
    
    # Fundamentals
    with tabs[0]:
        st.header("AI/ML Fundamentals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_video("https://www.youtube.com/watch?v=ukzFI9rgwfU", "Google - A beginner-friendly introduction")
            
        with col2:
            render_video("https://www.youtube.com/watch?v=4RixMPF4xis", "IBM Technology - AI vs ML vs DL")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("How ChatGPT Works")
            st.video("https://www.youtube.com/watch?v=VMj-3S1tku0")
            st.caption("3Blue1Brown - Neural Networks series (Part 1)")
            
        with col2:
            st.subheader("The History of AI")
            st.video("https://www.youtube.com/watch?v=G2fqAlgmoPo")
            st.caption("ColdFusion - AI History Documentary")
    
    # Machine Learning
    with tabs[1]:
        st.header("Machine Learning Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Linear Regression Explained")
            st.video("https://www.youtube.com/watch?v=nk2CQITm_eo")
            st.caption("StatQuest - Brilliant visual explanation")
            
        with col2:
            st.subheader("Decision Trees")
            st.video("https://www.youtube.com/watch?v=7VeUPuFGJHk")
            st.caption("StatQuest - Decision Trees clearly explained")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("K-Means Clustering")
            st.video("https://www.youtube.com/watch?v=4b5d3muPQmA")
            st.caption("StatQuest - Clustering demystified")
            
        with col2:
            st.subheader("Random Forest")
            st.video("https://www.youtube.com/watch?v=J4Wdy0Wc_xQ")
            st.caption("StatQuest - Random Forests explained")
    
    # Deep Learning
    with tabs[2]:
        st.header("Deep Learning & Neural Networks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("But what is a Neural Network?")
            st.video("https://www.youtube.com/watch?v=aircAruvnKk")
            st.caption("3Blue1Brown - The BEST neural network intro")
            
        with col2:
            st.subheader("Backpropagation")
            st.video("https://www.youtube.com/watch?v=Ilg3gGewQ5U")
            st.caption("3Blue1Brown - How neural nets learn")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Convolutional Neural Networks (CNN)")
            st.video("https://www.youtube.com/watch?v=FmpDIaiMIeA")
            st.caption("3Blue1Brown - CNNs visualized")
            
        with col2:
            st.subheader("Transformers Explained")
            st.video("https://www.youtube.com/watch?v=TQQlZhbC5ps")
            st.caption("Andrej Karpathy - Let's build GPT from scratch")
    
    # Generative AI
    with tabs[3]:
        st.header("Generative AI & LLMs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Attention Is All You Need")
            st.video("https://www.youtube.com/watch?v=SZorAJ4I-sA")
            st.caption("Yannic Kilcher - Paper explained")
            
        with col2:
            st.subheader("How GPT-3 Works")
            st.video("https://www.youtube.com/watch?v=MQnJZuBGmSQ")
            st.caption("Jay Alammar - Visual walkthrough")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prompt Engineering Masterclass")
            st.video("https://www.youtube.com/watch?v=dOxUroR57xs")
            st.caption("DeepLearning.AI - ChatGPT Prompt Engineering")
            
        with col2:
            st.subheader("RAG Explained")
            st.video("https://www.youtube.com/watch?v=T-D1OfcDW1M")
            st.caption("IBM - Retrieval Augmented Generation")
            
    # Developers Track
    with tabs[4]:
        st.header("💻 AI Engineering & Development")
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_video("https://www.youtube.com/watch?v=dyGTdb7qYKQ", "FreeCodeCamp - Full LangChain Course")
            
        with col2:
            render_video("https://www.youtube.com/watch?v=QAwV2V91y2U", "Simplilearn - AI Agents Explained")
            
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_video("https://www.youtube.com/watch?v=klTvEwg3oJk", "Fireship - Vector DBs in 100s")
            
        with col2:
            render_video("https://www.youtube.com/watch?v=Lcn9Ngj0QNA", "Alejandro AO - RAG Chatbot From Scratch")

    # Testers Track
    with tabs[5]:
        st.header("🧪 AI Testing, QA & Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_video("https://www.youtube.com/watch?v=mnJO57Xg7gw", "Edureka - RAG Tutorial")
            
        with col2:
            render_video("https://www.youtube.com/watch?v=l_6j_8_0-w4", "Google - Secure AI Framework")
            
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_video("https://www.youtube.com/watch?v=zJg5b25qL6c", "Computerphile - Prompt Injection")
            
        with col2:
            render_video("https://www.youtube.com/watch?v=5xvNIPc24jo", "Google Cloud - AI Fairness")
    
    # Career
    with tabs[6]:
        st.header("Career in AI/ML")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("How to Get into AI/ML")
            st.video("https://www.youtube.com/watch?v=5q87K1WaoFI")
            st.caption("Daniel Bourke - Complete roadmap")
            
        with col2:
            st.subheader("Day in the Life: ML Engineer")
            st.video("https://www.youtube.com/watch?v=JEq7o3DZNuw")
            st.caption("What ML engineers actually do")
        
        st.markdown("---")
        
        st.subheader("📚 Recommended Courses")
        
        courses = [
            {"name": "Machine Learning Specialization", "provider": "Coursera (Andrew Ng)", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "difficulty": "Beginner"},
            {"name": "Deep Learning Specialization", "provider": "Coursera (Andrew Ng)", "url": "https://www.coursera.org/specializations/deep-learning", "difficulty": "Intermediate"},
            {"name": "fast.ai Practical Deep Learning", "provider": "fast.ai", "url": "https://www.fast.ai/", "difficulty": "Intermediate"},
            {"name": "CS229: Machine Learning", "provider": "Stanford (Free)", "url": "https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU", "difficulty": "Advanced"},
        ]
        
        for course in courses:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**[{course['name']}]({course['url']})**")
            with col2:
                st.caption(course['provider'])
            with col3:
                if course['difficulty'] == "Beginner":
                    st.success(course['difficulty'])
                elif course['difficulty'] == "Intermediate":
                    st.warning(course['difficulty'])
                else:
                    st.error(course['difficulty'])
    
    # Playlists
    st.markdown("---")
    st.subheader("📺 Complete Playlists")
    
    playlists = [
        {"name": "3Blue1Brown - Neural Networks", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", "videos": 4},
        {"name": "StatQuest - Machine Learning", "url": "https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF", "videos": "100+"},
        {"name": "Andrej Karpathy - Neural Networks: Zero to Hero", "url": "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ", "videos": 8},
        {"name": "Sentdex - Machine Learning with Python", "url": "https://www.youtube.com/playlist?list=PLQVvV5Y6nv56qIJGLrhym5dpNW9jSmZzU", "videos": 72},
        {"name": "LangChain - AI Engineering", "url": "https://www.youtube.com/@LangChain", "videos": "Official"},
    ]
    
    for pl in playlists:
        st.markdown(f"🎬 [{pl['name']}]({pl['url']}) - {pl['videos']} videos")
