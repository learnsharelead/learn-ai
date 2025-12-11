import streamlit as st

def show():
    st.title("⚡ Performance Testing LLMs")
    
    st.markdown("""
    ### Speed & Scale
    
    AI Apps are slow. Latency kills UX. Learn how to benchmark and optimize
    throughput (Tokens/Second) and latency (Time to First Token).
    """)
    
    tabs = st.tabs([
        "⏱️ Key Metrics",
        "🏋️ Load Testing",
        "⚡ Optimization"
    ])
    
    # TAB 1: Metrics
    with tabs[0]:
        st.header("⏱️ Key Performance Indicators")
        
        metrics = [
            ("TTFT (Time To First Token)", "How long until the user sees the first word? (Target: <0.5s)"),
            ("TPOT (Time Per Output Token)", "How fast does the text stream? (Target: <50ms)"),
            ("End-to-End Latency", "Total time for full response."),
            ("RPS (Requests Per Second)", "How many concurrent users can you handle?"),
            ("Error Rate", "Percentage of failed/timeout requests."),
        ]
        
        for name, desc in metrics:
            st.metric(name, desc)

    # TAB 2: Load Testing
    with tabs[1]:
        st.header("🏋️ Load Testing Tools")
        
        st.subheader("Locust")
        st.markdown("Python-based load testing tool. Script user behavior.")
        
        st.code('''
from locust import HttpUser, task, between

class AIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def chat(self):
        self.client.post("/chat", json={
            "message": "Tell me a joke"
        })
        ''', language="python")
        
        st.subheader("LLMPerf")
        st.markdown("Specific tool for benchmarking LLM providers (Tokens/sec).")

    # TAB 3: Optimization
    with tabs[2]:
        st.header("⚡ Optimization Strategies")
        
        st.markdown("""
        **1. Streaming:** ALWAYS stream responses. Don't make users wait for the full generation.
        
        **2. Smaller Models:** Llama 3 8B is much faster than 70B.
        
        **3. Groq (LPU):** Specialized hardware for massive speed (500+ tokens/sec).
        
        **4. Speculative Decoding:** Use a small model to draft, large model to verify.
        
        **5. VLLM:** Optimized inference engine for hosting your own models.
        """)
