import streamlit as st
import plotly.graph_objects as go

def show():
    st.title("🤖 Module 1: What is AI?")
    
    st.markdown("""
    ### 🎯 Simple explanations + Formal definitions for your notes!
    """)
    
    tabs = st.tabs(["🌟 AI Explained Simply", "📚 Definitions & Notes", "🏠 AI in Your Life", "🎮 Try It Yourself", "📺 How ChatGPT Works"])
    
    # TAB 1: Simple Explanation
    with tabs[0]:
        st.header("🌟 What is AI? (The Simple Version)")
        
        st.markdown("""
        ### Imagine Teaching a Child...
        
        Think about how you learned to recognize a **cat** when you were little:
        
        1. 👶 Your parents showed you MANY cats
        2. 📸 Big cats, small cats, orange cats, black cats
        3. 🧠 Your brain found patterns: *"4 legs, whiskers, pointy ears, meows"*
        4. ✅ Now you can recognize ANY cat, even ones you've never seen!
        
        **That's exactly how AI learns!** 🎉
        """)
        
        st.markdown("---")
        
        # Side by side: Simple + Formal
        st.subheader("🍕 The Analogy vs 📘 The Definition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **AI = Computer that can learn and think**
            
            Like a really smart assistant that:
            - Learns from examples
            - Gets better over time
            - Never forgets
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Artificial Intelligence (AI)**: The branch of computer 
            science concerned with building machines capable of 
            performing tasks that typically require human intelligence,
            such as visual perception, speech recognition, and 
            decision-making.
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **ML = Teaching computers by showing examples**
            
            Instead of writing rules like:
            "IF email has 'FREE' THEN spam"
            
            You show 10,000 spam emails and 
            let the computer figure out the rules!
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Machine Learning (ML)**: A subset of AI that provides 
            systems the ability to automatically learn and improve 
            from experience without being explicitly programmed. 
            ML focuses on developing algorithms that can access 
            data and use it to learn for themselves.
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **Deep Learning = Brain-inspired learning**
            
            Artificial "neurons" in layers, like your brain!
            - First layer: spots edges
            - Next layer: spots shapes
            - Final layer: recognizes faces!
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Deep Learning (DL)**: A specialized subset of Machine 
            Learning based on artificial neural networks with 
            multiple layers (hence "deep"). Deep learning models 
            can learn to represent data with multiple levels of 
            abstraction, automatically discovering features needed 
            for classification or prediction.
            """)
    
    # TAB 2: Definitions & Notes
    with tabs[1]:
        st.header("📚 Complete Definitions (For Notes)")
        
        st.markdown("""
        ### 📋 Copy These for Your Notes!
        """)
        
        st.subheader("🔑 Key Definitions")
        
        definitions = {
            "Artificial Intelligence (AI)": "The simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
            
            "Machine Learning (ML)": "A subset of AI that enables systems to learn and improve from experience automatically without being explicitly programmed. It focuses on developing algorithms that can access data and learn from it.",
            
            "Deep Learning (DL)": "A subset of Machine Learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input.",
            
            "Neural Network": "A computing system inspired by biological neural networks in the human brain. It consists of interconnected nodes (neurons) that process information using connectionist approaches.",
            
            "Training Data": "The dataset used to teach a machine learning model. It contains examples with known inputs and outputs that the model learns from.",
            
            "Model": "A mathematical representation of a real-world process. In ML, it's the algorithm that has been trained on data to make predictions.",
            
            "Prediction": "The output of a trained model when given new input data. It represents the model's best guess based on patterns learned during training.",
            
            "Feature": "An individual measurable property of the data being observed. Features are the input variables used to make predictions.",
            
            "Label": "The output variable or target that the model is trying to predict. In supervised learning, labels are provided in the training data.",
            
            "Algorithm": "A set of rules or instructions given to the computer to help it learn from data. Examples: Linear Regression, Decision Trees, Neural Networks.",
        }
        
        for term, definition in definitions.items():
            with st.expander(f"📖 {term}"):
                st.markdown(f"**{term}**")
                st.markdown(definition)
                st.code(f"{term}: {definition}", language="text")
        
        st.markdown("---")
        
        st.subheader("📊 The AI Hierarchy")
        
        st.graphviz_chart("""
        digraph Hierarchy {
            rankdir=TB;
            node [shape=box, style=filled];
            
            AI [label="Artificial Intelligence\\n(Broadest concept)", fillcolor=lightblue];
            ML [label="Machine Learning\\n(Learning from data)", fillcolor=lightgreen];
            DL [label="Deep Learning\\n(Neural networks)", fillcolor=orange];
            
            AI -> ML [label="subset of"];
            ML -> DL [label="subset of"];
        }
        """)
        
        st.info("""
        **Remember:** 
        - All Deep Learning is Machine Learning
        - All Machine Learning is AI
        - But NOT all AI is Machine Learning!
        """)
        
        st.subheader("📝 Types of Machine Learning")
        
        learning_types = {
            "Supervised Learning": {
                "definition": "Learning with labeled examples (input-output pairs). The model learns to map inputs to correct outputs.",
                "analogy": "Like a student with a textbook that has answers in the back",
                "examples": "Spam detection, house price prediction, image classification"
            },
            "Unsupervised Learning": {
                "definition": "Learning without labels. The model finds hidden patterns or structures in data.",
                "analogy": "Like sorting a pile of clothes without being told the categories",
                "examples": "Customer segmentation, anomaly detection, topic modeling"
            },
            "Reinforcement Learning": {
                "definition": "Learning through trial and error by receiving rewards or penalties for actions.",
                "analogy": "Like training a dog with treats - good behavior gets rewards",
                "examples": "Game playing (AlphaGo), robotics, self-driving cars"
            },
        }
        
        for name, info in learning_types.items():
            with st.expander(f"🎯 {name}"):
                st.markdown(f"**📘 Definition:** {info['definition']}")
                st.markdown(f"**🍕 Analogy:** {info['analogy']}")
                st.markdown(f"**💡 Examples:** {info['examples']}")
    
    # TAB 3: AI in Daily Life
    with tabs[2]:
        st.header("🏠 AI You Already Use Every Day!")
        
        st.markdown("### You've been using AI without knowing it! 🤯")
        
        ai_examples = [
            ("📱 Face Unlock", "Computer Vision + Deep Learning", "Uses facial recognition neural networks to identify your unique facial features", "Trained on millions of faces to learn what makes each person unique"),
            ("🗣️ Siri/Google Assistant", "Natural Language Processing (NLP)", "Converts speech to text, understands intent, generates response", "Uses transformers/neural networks trained on billions of conversations"),
            ("📧 Spam Filter", "Classification (Supervised Learning)", "Classifies emails as spam or not based on content patterns", "Trained on millions of labeled emails to recognize spam characteristics"),
            ("🎵 Spotify Discover", "Recommendation System", "Collaborative & content filtering to suggest music", "Analyzes listening patterns of millions of users with similar tastes"),
            ("🚗 Google Maps ETA", "Regression + Time Series", "Predicts arrival time based on traffic patterns", "Uses historical and real-time data from millions of drivers"),
            ("📱 Autocorrect", "NLP + Language Model", "Predicts intended words and next words", "Trained on billions of text messages to learn language patterns"),
        ]
        
        for name, category, simple, technical in ai_examples:
            with st.expander(f"{name}"):
                st.markdown(f"**📘 AI Category:** {category}")
                st.markdown(f"**🍕 Simple:** {simple}")
                st.markdown(f"**🔬 Technical:** {technical}")
    
    # TAB 4: Try It Yourself
    with tabs[3]:
        st.header("🎮 Be the AI! (Interactive Game)")
        
        st.markdown("""
        ### Can YOU spot the pattern like an AI?
        """)
        
        # Game 1: Spam Detection
        st.subheader("Game 1: Spam Detection")
        
        with st.expander("📧 Training Examples (Click to see)"):
            st.markdown("""
            | Email Subject | Label |
            |--------------|-------|
            | "FREE IPHONE CLICK NOW!!!" | 🚫 SPAM |
            | "Meeting tomorrow at 3pm" | ✅ Normal |
            | "CONGRATULATIONS YOU WON $1,000,000" | 🚫 SPAM |
            | "Dinner plans for Saturday?" | ✅ Normal |
            | "ACT NOW! LIMITED TIME OFFER!!!" | 🚫 SPAM |
            | "Report attached for review" | ✅ Normal |
            """)
        
        user_guess = st.text_input("What patterns do SPAM emails have?")
        
        if user_guess:
            st.success("""
            ### Great thinking! 🧠
            
            **Patterns AI finds in spam:**
            - ALL CAPS words
            - Multiple exclamation marks!!!
            - Words like "FREE", "WINNER", "CLICK NOW"
            - Urgency phrases ("ACT NOW", "LIMITED TIME")
            
            **You just thought like an AI!** 🎉
            """)
            
            st.info("""
            **📘 Technical Note:** This is called "Feature Engineering" - 
            identifying which characteristics (features) of the data are 
            useful for making predictions.
            """)
    
    # TAB 5: ChatGPT Story
    with tabs[4]:
        st.header("📺 How ChatGPT Learned to Talk")
        
        st.subheader("🍕 The Simple Story")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            ### 📚 Step 1: Reading
            
            **ChatGPT read the internet**
            
            - Books, Wikipedia
            - Websites, forums
            - Code, conversations
            
            *100,000+ books worth!*
            """)
            
        with col2:
            st.info("""
            ### 🔮 Step 2: Guessing
            
            **Predict the next word**
            
            "The sky is..." → blue
            "2 + 2 =" → 4
            
            *Billions of predictions!*
            """)
            
        with col3:
            st.info("""
            ### 👍 Step 3: Feedback
            
            **Humans rated responses**
            
            👍 Helpful = Good
            👎 Harmful = Bad
            
            *Learned to be helpful!*
            """)
        
        st.markdown("---")
        
        st.subheader("📘 The Technical Details (For Notes)")
        
        with st.expander("🔬 Pre-training Phase"):
            st.markdown("""
            **Large Language Model (LLM) Pre-training:**
            
            - **Data:** ~570 GB of text (300 billion+ tokens)
            - **Objective:** Next Token Prediction (Causal Language Modeling)
            - **Architecture:** Transformer (decoder-only, like GPT)
            - **Parameters:** 175B (GPT-3) to estimated 1.7T (GPT-4)
            
            **Loss Function:** Cross-entropy loss, minimizing the difference 
            between predicted and actual next tokens.
            """)
        
        with st.expander("🔬 RLHF (Reinforcement Learning from Human Feedback)"):
            st.markdown("""
            **Fine-tuning with Human Preferences:**
            
            1. **Supervised Fine-Tuning (SFT):** Train on human-written responses
            2. **Reward Model:** Humans rank response quality
            3. **PPO Training:** Optimize model to maximize reward
            
            **Purpose:** Align model behavior with human values 
            (helpful, harmless, honest).
            """)
        
        with st.expander("🔬 Transformer Architecture"):
            st.markdown("""
            **Key Components:**
            
            - **Self-Attention:** Each token attends to all other tokens
            - **Multi-Head Attention:** Multiple attention "perspectives"
            - **Feed-Forward Networks:** Process attended information
            - **Positional Encoding:** Inject position information
            
            **Formula (Attention):**
            
            Attention(Q, K, V) = softmax(QK^T / √d_k) V
            
            Where Q=Query, K=Key, V=Value, d_k=dimension
            """)
        
        st.success("""
        ### 💡 Key Takeaway
        
        **Simple:** ChatGPT is the world's best autocomplete!
        
        **Technical:** It's a decoder-only transformer trained with 
        causal language modeling, then aligned with RLHF.
        
        *Both are correct - just different levels of detail!*
        """)
