import streamlit as st

def show():
    st.title("📈 A/B Testing Prompts")
    
    st.markdown("""
    ### Data-Driven Prompt Engineering
    
    Stop guessing. Run statistical experiments to prove which prompt performs better.
    """)
    
    tabs = st.tabs([
        "🧪 Experiment Design",
        "📊 Implementation",
        "🔢 Significance"
    ])
    
    # TAB 1: Design
    with tabs[0]:
        st.header("🧪 Designing an A/B Test")
        
        st.markdown("""
        1. **Hypothesis:** "Adding 'think step by step' will improve math accuracy."
        2. **Variants:**
           - **Control (A):** "Solve 2x+5=10"
           - **Variant (B):** "Solve 2x+5=10. Think step by step."
        3. **Metric:** Accuracy (did it get x=2.5?)
        4. **Sample Size:** Need at least 50-100 runs to be significant.
        """)

    # TAB 2: Implementation
    with tabs[1]:
        st.header("📊 Running the Experiment")
        
        st.markdown("**Promptfoo** is the best tool for this.")
        
        st.code('''
# promptfooconfig.yaml
prompts:
  - "Write a haiku about {topic}"
  - "Write a haiku about {topic}, do not count syllables explicitly"

providers: [openai:gpt-3.5-turbo]

tests:
  - vars:
      topic: cats
  - vars:
      topic: dogs
        ''', language="yaml")
        
        st.code("npx promptfoo eval", language="bash")
        
        st.image("https://github.com/promptfoo/promptfoo/raw/main/assets/screenshot-matrix.png", caption="Promptfoo Matrix View")

    # TAB 3: Significance
    with tabs[2]:
        st.header("🔢 Statistical Significance")
        
        with st.expander("Why it matches"):
            st.markdown("""
            If A gets 80% and B gets 82% on 10 examples, it's just noise.
            
            Use a **Chi-Squared Test** or **T-Test** to confirm if the difference 
            is real or luck.
            """)
        
        st.success("Rule of Thumb: If you can't test >100 examples, don't trust small % improvements.")
