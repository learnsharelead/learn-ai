import streamlit as st
import numpy as np
import plotly.graph_objects as go

def show():
    st.title("🤖 Module 8: ChatGPT & Generative AI")
    
    st.markdown("""
    ### How AI Creates Text, Images, and More!
    
    *Finally understand how ChatGPT actually works - explained simply!*
    """)
    
    tabs = st.tabs([
        "🎯 What is Generative AI?",
        "💬 How ChatGPT Works",
        "🎨 How Image AI Works",
        "✍️ Talking to AI (Prompts)",
        "🎮 Try It Yourself"
    ])
    
    # TAB 1: What is Generative AI
    with tabs[0]:
        st.header("🎯 What is Generative AI?")
        
        st.markdown("""
        ### The Artist vs The Critic
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            ### 🔍 Traditional AI (The Critic)
            
            **Analyzes and classifies things**
            
            - Is this email spam? → Yes/No
            - What animal is this? → Cat
            - Is this movie review positive? → Yes
            
            *It judges, doesn't create*
            """)
            
        with col2:
            st.success("""
            ### 🎨 Generative AI (The Artist)
            
            **Creates new content**
            
            - Write me a poem about cats
            - Draw a sunset over mountains
            - Compose a happy song
            
            *It creates, not just judges!*
            """)
        
        st.markdown("---")
        
        st.subheader("🌟 Examples of Generative AI")
        
        examples = [
            ("💬 ChatGPT", "Creates text", "Write anything: stories, code, emails, recipes"),
            ("🎨 DALL-E / Midjourney", "Creates images", "Draw any picture from a description"),
            ("🎵 Suno / Udio", "Creates music", "Compose songs in any style"),
            ("🎬 Sora", "Creates videos", "Generate realistic videos from text"),
            ("🗣️ ElevenLabs", "Creates voice", "Clone any voice, speak any language"),
        ]
        
        for name, what, desc in examples:
            with st.expander(name):
                st.markdown(f"**What it does:** {what}")
                st.markdown(f"**Example:** {desc}")
        
        st.success("""
        ### 💡 The Big Idea
        
        Generative AI learned from BILLIONS of examples (text, images, etc.)
        and can now **create new** content that looks like it was made by humans!
        
        *Like an art student who studied every painting, can now paint new ones.*
        """)
    
    # TAB 2: How ChatGPT Works
    with tabs[1]:
        st.header("💬 How ChatGPT Actually Works")
        
        st.markdown("""
        ### The World's Best Autocomplete 📱
        
        You know how your phone suggests the next word when texting?
        
        **ChatGPT is basically that, but 1000x smarter!**
        """)
        
        st.subheader("Step 1: Predict the Next Word")
        
        st.info("""
        ### 🔮 The Core Skill
        
        ChatGPT only does ONE thing: **Predict what comes next**
        
        "The sky is..." → **"blue"** (most likely)
        "I love eating..." → **"pizza"** or **"food"**
        "To be or not to..." → **"be"**
        
        That's it! But it does this REALLY well.
        """)
        
        st.markdown("---")
        
        st.subheader("Step 2: Chain the Predictions")
        
        st.markdown("""
        **Prompt:** "Write a haiku about cats"
        
        **ChatGPT's brain:**
        
        1. "Write a haiku about cats" + → **"Soft"**
        2. "Write a haiku about cats Soft" + → **"paws"**
        3. "Write a haiku about cats Soft paws" + → **"on"**
        4. Continue until done...
        
        **Result:** "Soft paws on the floor, Whiskers twitch in morning light, Nap time calls again"
        """)
        
        st.success("""
        ### 💡 Key Insight
        
        ChatGPT doesn't "think" or "understand" like humans.
        
        It's incredibly good at **predicting what text should come next**
        based on patterns from reading the entire internet.
        """)
        
        st.markdown("---")
        
        st.subheader("Step 3: The Training Journey")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📚 Phase 1: Reading
            
            **Read almost everything:**
            - All of Wikipedia
            - Millions of books
            - Websites, forums
            - Code repositories
            
            *Like studying for the world's biggest test*
            """)
            
        with col2:
            st.markdown("""
            ### 📝 Phase 2: Practice
            
            **Predict billions of words:**
            
            "The ___ is blue" → sky
            "Paris is in ___" → France
            "def hello(): ___" → return
            
            *Get better at predicting*
            """)
            
        with col3:
            st.markdown("""
            ### 👍 Phase 3: Feedback
            
            **Humans rated responses:**
            
            👍 Helpful, honest, harmless
            👎 Wrong, rude, dangerous
            
            *Learn to be a good assistant*
            """)
        
        st.info("""
        ### 🧠 The Numbers Are Mind-Blowing
        
        - **Training data:** ~570 GB of text (300 billion words!)
        - **Parameters:** 175 billion (GPT-3) to 1.7 trillion (GPT-4)
        - **Training cost:** ~$100 million
        - **Training time:** Months on thousands of GPUs
        """)
    
    # TAB 3: Image AI
    with tabs[2]:
        st.header("🎨 How AI Creates Images")
        
        st.markdown("""
        ### The Noise-to-Image Magic ✨
        
        Imagine turning TV static into a beautiful painting!
        """)
        
        st.subheader("The Diffusion Process")
        
        st.markdown("""
        **Step 1: Start with random noise (like TV static)**
        
        ```
        ░░░▒▒░░▓▓░░▒░░▓
        ▒▓░░▒▒░░▓░░▒░▒░
        ░░▓▓░░▒▒░░▓░░▒▒
        ```
        
        **Step 50: AI slowly removes noise**
        
        ```
        ░░░   .   ▓▓   .
        ▒     🌅      ░
        ░░         ▒▒░░
        ```
        
        **Step 100: Clear image appears!**
        
        ```
        ☀️ = = = = = = =
        🌄🌄🌄🌄🌄🌄🌄🌄
        🌊🌊🌊🌊🌊🌊🌊🌊
        ```
        """)
        
        st.success("""
        ### 💡 Simple Explanation
        
        1. **Training:** AI learned to remove noise from millions of images
        2. **Generation:** Start with pure noise + your description
        3. **De-noise:** Slowly remove noise, guided by your description
        4. **Result:** Beautiful image matching your words!
        
        *Like sculpting: start with marble block, remove what doesn't belong*
        """)
        
        st.markdown("---")
        
        st.subheader("🎨 The Magic of Text-to-Image")
        
        st.markdown("""
        **Your Prompt:** "A cat wearing a tiny hat, sitting on a throne"
        
        **AI's Process:**
        
        1. 🐱 "Cat" → Add cat-like features
        2. 👒 "Tiny hat" → Add hat on head
        3. 👑 "Throne" → Add royal throne
        4. 🎨 Style it → Make it look realistic/artistic
        
        **50-100 steps later:** Beautiful cat emperor! 👑🐱
        """)
        
        st.info("""
        ### 🔥 Popular Image AI Tools
        
        - **DALL-E 3** (by OpenAI) - Best for variety
        - **Midjourney** - Best for artistic images
        - **Stable Diffusion** - Free and open source
        - **Adobe Firefly** - Built into Photoshop
        """)
    
    # TAB 4: Prompts
    with tabs[3]:
        st.header("✍️ How to Talk to AI (Prompt Engineering)")
        
        st.markdown("""
        ### The Better You Ask, The Better AI Answers!
        
        *Like giving directions: "Go somewhere" vs "Go to 123 Main Street"*
        """)
        
        st.subheader("❌ Bad Prompt vs ✅ Good Prompt")
        
        examples = [
            {
                "bad": "Write something about dogs",
                "good": "Write a 100-word children's story about a golden retriever puppy who learns to share toys",
                "why": "Specific length, audience, breed, and plot"
            },
            {
                "bad": "Help me with my email",
                "good": "Write a polite email to my boss asking for Friday off for a doctor's appointment. Keep it professional and brief.",
                "why": "Clear tone, purpose, and recipient context"
            },
            {
                "bad": "Make an image of nature",
                "good": "A serene mountain lake at sunset, with pine trees reflected in the water, in the style of a landscape painting",
                "why": "Specific subject, time, style, and composition"
            },
        ]
        
        for i, ex in enumerate(examples, 1):
            st.markdown(f"### Example {i}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.error(f"**❌ Vague:** {ex['bad']}")
                
            with col2:
                st.success(f"**✅ Clear:** {ex['good']}")
            
            st.info(f"**Why it's better:** {ex['why']}")
            st.markdown("---")
        
        st.subheader("🎯 Prompt Tips")
        
        tips = [
            ("Be Specific", "Include details about length, format, style, audience"),
            ("Give Context", "Explain WHY you need it and WHO it's for"),
            ("Use Examples", "Show AI what a good answer looks like"),
            ("Role Play", "Say 'Act as a chef' or 'You are a teacher'"),
            ("Break It Down", "Ask one thing at a time for complex tasks"),
        ]
        
        for title, desc in tips:
            st.markdown(f"**{title}:** {desc}")
    
    # TAB 5: Try It
    with tabs[4]:
        st.header("🎮 Practice Prompting!")
        
        st.markdown("""
        ### Improve These Prompts
        
        Take the vague prompt and make it better!
        """)
        
        st.subheader("Challenge 1: Recipe Request")
        
        st.error("**Original (vague):** Tell me how to cook dinner")
        
        improved1 = st.text_area(
            "Your improved prompt:",
            key="improve1",
            placeholder="Make it specific! What dish? For how many people? Any diet restrictions?"
        )
        
        if improved1:
            # Check for good elements
            good_elements = []
            if any(word in improved1.lower() for word in ['recipe', 'dish', 'meal']):
                good_elements.append("✅ Specified a dish type")
            if any(word in improved1.lower() for word in ['people', 'servings', 'person']):
                good_elements.append("✅ Mentioned serving size")
            if any(word in improved1.lower() for word in ['minutes', 'quick', 'easy', 'time']):
                good_elements.append("✅ Mentioned time/difficulty")
            if any(word in improved1.lower() for word in ['vegetarian', 'healthy', 'diet', 'allergies']):
                good_elements.append("✅ Mentioned dietary needs")
            
            if len(good_elements) >= 2:
                st.success(f"🎉 Great prompt! You included:\n" + "\n".join(good_elements))
            else:
                st.info("💡 Try adding: specific dish, serving size, time available, dietary needs")
        
        st.markdown("---")
        
        st.subheader("Challenge 2: Image Generation")
        
        st.error("**Original (vague):** Create a picture of a house")
        
        improved2 = st.text_area(
            "Your improved prompt:",
            key="improve2",
            placeholder="Add: style, time of day, surroundings, colors, mood!"
        )
        
        if improved2:
            good_elements = []
            if any(word in improved2.lower() for word in ['style', 'painting', 'photo', 'cartoon', 'realistic']):
                good_elements.append("✅ Specified art style")
            if any(word in improved2.lower() for word in ['sunset', 'night', 'morning', 'day']):
                good_elements.append("✅ Mentioned time of day")
            if any(word in improved2.lower() for word in ['garden', 'trees', 'mountains', 'city', 'snow']):
                good_elements.append("✅ Described surroundings")
            if any(word in improved2.lower() for word in ['cozy', 'modern', 'vintage', 'color', 'warm']):
                good_elements.append("✅ Added mood/details")
            
            if len(good_elements) >= 2:
                st.success(f"🎉 Great prompt! You included:\n" + "\n".join(good_elements))
            else:
                st.info("💡 Try adding: art style, time of day, surroundings, colors/mood")
        
        st.markdown("---")
        
        st.success("""
        ### 🎓 Prompt Engineering is a Skill!
        
        The more specific and clear you are, the better AI understands you.
        
        **Practice makes perfect!** Try different ways of asking until you get great results.
        """)
