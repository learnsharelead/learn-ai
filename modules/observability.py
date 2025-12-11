import streamlit as st

def show():
    st.title("📊 Observability & Monitoring")
    
    st.markdown("""
    ### Why did the AI say that?
    
    Production AI is a black box. **Observability tools** let you peek inside to debug, 
    trace execution, and monitor quality.
    """)
    
    tabs = st.tabs([
        "🔭 Tracing",
        "📈 Monitoring Tools",
        "👨‍💻 Code Implementation"
    ])
    
    # TAB 1: Tracing
    with tabs[0]:
        st.header("🔭 Execution Tracing")
        
        st.markdown("""
        **Tracing** visualizes the full chain of events:
        1. User Input
        2. Retriever (What docs were found?)
        3. Prompt Assembler (What exactly went to the LLM?)
        4. LLM Call (Latency, Tokens)
        5. Output Parser
        """)
        
        st.info("Without tracing, you are guessing why RAG failed.")

    # TAB 2: Tools
    with tabs[1]:
        st.header("📈 Top Tools")
        
        tools = [
            ("🦜 **LangSmith**", "Best for LangChain. Deep replays, dataset testing.", "Paid (Generous free tier)"),
            ("🏋️ **Weights & Biases**", "Experiment tracking (Prompts as hyperparameters).", "Enterprise standard"),
            ("🦅 **Phoenix (Arize)**", "Open source tracing and eval.", "Great for local dev"),
            ("🍯 **HoneyHive**", "Fine-tuning and prompt management.", "Startups"),
        ]
        
        for name, desc, cost in tools:
            st.markdown(f"**{name}**")
            st.markdown(f"- {desc}")
            st.caption(f"Cost: {cost}")
            st.markdown("---")

    # TAB 3: Code
    with tabs[2]:
        st.header("👨‍💻 Adding LangSmith Support")
        
        st.markdown("It's literally 3 environment variables.")
        
        st.code('''
# .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="nexus-ai-prod"
        ''', language="bash")
        
        st.code('''
from langchain.chat_models import ChatOpenAI

# Just run your code normally. 
# LangChain automatically logs everything to the dashboard!
llm = ChatOpenAI()
llm.invoke("Hello world")
        ''', language="python")
        
        st.success("Go to smith.langchain.com to see the trace!")
