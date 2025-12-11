import streamlit as st

def show():
    st.title("💰 AI Cost Optimization")
    
    st.markdown("""
    ### Don't Break the Bank
    
    LLM API costs can explode quickly. Learn how to track tokens, cache responses, 
    and select the right model to save 90% significantly.
    """)
    
    tabs = st.tabs([
        "💸 Cost Factors",
        "🧠 Caching Strategies",
        "📉 Model Selection",
        "🧮 Calculator"
    ])
    
    # TAB 1: Cost Factors
    with tabs[0]:
        st.header("💸 Understanding Costs")
        
        st.info("Pricing is usually per **1 Million Tokens** (Input vs Output).")
        
        st.markdown("""
        | Model | Input Cost / 1M | Output Cost / 1M |
        |-------|-----------------|------------------|
        | **GPT-4o** | \$2.50 | \$10.00 |
        | **GPT-4o-mini** | \$0.15 | \$0.60 |
        | **Claude 3.5 Sonnet** | \$3.00 | \$15.00 |
        | **Gemini 1.5 Flash** | \$0.075 | \$0.30 |
        | **Llama 3 (Groq)** | ~\$0.10 | ~\$0.10 |
        """)
        
        st.warning("""
        **Hidden Costs:**
        1. **RAG Context:** Sending 10 documents per query = massive input costs.
        2. **Chain-of-Thought:** Asking model to "think step by step" = massive output costs.
        3. **Agents:** Agents loops can run dozens of times per user request.
        """)

    # TAB 2: Caching
    with tabs[1]:
        st.header("🧠 Caching Strategies")
        
        st.markdown("""
        **The #1 Rule:** Never pay for the same answer twice.
        """)
        
        st.subheader("1. Exact Match Caching")
        st.markdown("Hash the prompt. If seen before, return stored answer.")
        
        st.subheader("2. Semantic Caching (GPTCache)")
        st.markdown("If a new query is *similar* to an old one, return the old answer.")
        
        st.code('''
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()

# First call: Hits API (Slow + Cost)
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'What is GitHub?'}]
)

# Second call: Hits Cache (Fast + Free)
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'What is GitHub?'}]
)
        ''', language="python")

    # TAB 3: Selection
    with tabs[2]:
        st.header("📉 Model Selection Strategy")
        
        st.markdown("""
        **The Cascade Pattern:**
        Try the cheapest model first. If it fails (or is unsure), try the expensive one.
        """)
        
        st.code('''
def intelligent_router(query):
    # 1. Try Free/Cheap model (Llama 3 8B or GPT-4o-mini)
    response = cheap_model.generate(query)
    
    # 2. Check confidence/quality
    if check_quality(response) == "GOOD":
        return response
        
    # 3. Fallback to Smart model (GPT-4o/Claude 3.5)
    print("Escalating to GPT-4o...")
    return expensive_model.generate(query)
        ''', language="python")
        
        st.success("Result: 80% of queries handled by cheap model. 5x cost reduction.")

    # TAB 4: Calculator
    with tabs[3]:
        st.header("🧮 Token Cost Calculator")
        
        model = st.selectbox("Model", ["GPT-4o", "GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 Flash"])
        
        col1, col2 = st.columns(2)
        avg_input = col1.number_input("Avg Input Tokens", value=1000)
        avg_output = col2.number_input("Avg Output Tokens", value=500)
        daily_reqs = st.number_input("Daily Requests", value=1000)
        
        # Approximate pricing (Nov 2024)
        prices = {
            "GPT-4o": (2.5, 10.0),
            "GPT-4o-mini": (0.15, 0.60),
            "Claude 3.5 Sonnet": (3.0, 15.0),
            "Gemini 1.5 Flash": (0.075, 0.30)
        }
        
        in_p, out_p = prices[model]
        
        daily_cost = (daily_reqs * ((avg_input/1e6)*in_p + (avg_output/1e6)*out_p))
        monthly_cost = daily_cost * 30
        
        st.metric("Estimated Monthly Cost", f"${monthly_cost:,.2f}")
        
        if monthly_cost > 1000:
            st.error("Too expensive! Consider caching or using a smaller model.")
        else:
            st.success("Reasonable budget!")
