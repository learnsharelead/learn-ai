import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

def show():
    st.title("⚡ Performance Testing for AI Systems")
    
    st.markdown("""
    **Performance is a feature.** Slow AI apps lose users. Learn how to benchmark, optimize, 
    and scale your LLM applications for production.
    """)
    
    tabs = st.tabs([
        "⏱️ Key Metrics",
        "🧮 Latency Calculator",
        "🏋️ Load Testing",
        "📊 Benchmarks",
        "⚡ Optimization",
        "🔔 Monitoring"
    ])
    
    # TAB 1: Key Metrics
    with tabs[0]:
        st.header("⏱️ Performance Metrics Explained")
        
        st.markdown("""
        Understanding these metrics is crucial for building production-ready AI systems.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Latency Metrics")
            
            metrics = {
                "TTFT (Time To First Token)": {
                    "desc": "How long until the user sees the first word?",
                    "target": "< 500ms",
                    "why": "Users perceive instant feedback as responsive"
                },
                "TPOT (Time Per Output Token)": {
                    "desc": "How fast does the text stream?",
                    "target": "< 50ms",
                    "why": "Smooth streaming feels natural"
                },
                "E2E Latency": {
                    "desc": "Total time for complete response",
                    "target": "< 5s for 500 tokens",
                    "why": "Users abandon after 10s"
                }
            }
            
            for name, info in metrics.items():
                with st.expander(f"**{name}**"):
                    st.markdown(f"**Description**: {info['desc']}")
                    st.success(f"**Target**: {info['target']}")
                    st.info(f"**Why it matters**: {info['why']}")
        
        with col2:
            st.subheader("Throughput Metrics")
            
            throughput = {
                "Tokens/Second": {
                    "desc": "Generation speed",
                    "target": "> 20 tokens/s",
                    "why": "Faster = cheaper at scale"
                },
                "RPS (Requests/Second)": {
                    "desc": "Concurrent user capacity",
                    "target": "> 100 RPS",
                    "why": "Scalability requirement"
                },
                "Error Rate": {
                    "desc": "Failed/timeout requests",
                    "target": "< 0.1%",
                    "why": "Reliability is critical"
                }
            }
            
            for name, info in throughput.items():
                with st.expander(f"**{name}**"):
                    st.markdown(f"**Description**: {info['desc']}")
                    st.success(f"**Target**: {info['target']}")
                    st.info(f"**Why it matters**: {info['why']}")
        
        st.markdown("---")
        st.markdown("""
        ### 📐 The Formula
        
        **Total Latency** = TTFT + (Number of Tokens × TPOT)
        
        **Example**: 
        - TTFT = 300ms
        - TPOT = 40ms
        - Output = 100 tokens
        - **Total** = 300ms + (100 × 40ms) = **4.3 seconds**
        """)

    # TAB 2: Latency Calculator
    with tabs[1]:
        st.header("🧮 Interactive Latency Calculator")
        
        st.markdown("Model your expected latency based on different configurations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            ttft = st.slider("TTFT (ms)", 100, 2000, 500, 50)
            tpot = st.slider("TPOT (ms)", 10, 200, 50, 10)
            tokens = st.slider("Output Tokens", 50, 1000, 200, 50)
            
        with col2:
            st.subheader("Calculated Metrics")
            
            total_latency = ttft + (tokens * tpot)
            tokens_per_sec = 1000 / tpot if tpot > 0 else 0
            
            st.metric("Total Latency", f"{total_latency:,} ms", 
                     delta=f"{total_latency/1000:.1f}s")
            st.metric("Tokens/Second", f"{tokens_per_sec:.1f}")
            st.metric("Words/Second", f"{tokens_per_sec * 0.75:.1f}")
            
            # User experience rating
            if total_latency < 2000:
                st.success("✅ Excellent UX - Users will love this!")
            elif total_latency < 5000:
                st.warning("⚠️ Acceptable - Consider optimization")
            else:
                st.error("❌ Too Slow - Users will abandon")
        
        # Visualization
        token_range = range(50, 1001, 50)
        latencies = [ttft + (t * tpot) for t in token_range]
        
        fig = px.line(
            x=list(token_range),
            y=latencies,
            labels={'x': 'Output Tokens', 'y': 'Total Latency (ms)'},
            title='Latency vs Output Length'
        )
        fig.add_hline(y=5000, line_dash="dash", line_color="red", 
                     annotation_text="5s threshold")
        
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Load Testing
    with tabs[2]:
        st.header("🏋️ Load Testing with Locust")
        
        st.markdown("""
        **Load testing** simulates real-world traffic to find your breaking point.
        """)
        
        st.subheader("1. Install Locust")
        st.code("pip install locust", language="bash")
        
        st.subheader("2. Create Test Script")
        st.code('''
# locustfile.py
from locust import HttpUser, task, between
import random

class AIAppUser(HttpUser):
    wait_time = between(1, 3)  # Simulate user think time
    
    @task(3)  # 75% of requests
    def chat_short(self):
        """Simulate short chat messages"""
        self.client.post("/api/chat", json={
            "message": "What is machine learning?",
            "max_tokens": 100
        })
    
    @task(1)  # 25% of requests
    def chat_long(self):
        """Simulate long-form generation"""
        self.client.post("/api/chat", json={
            "message": "Write a detailed explanation of transformers",
            "max_tokens": 500
        })
    
    def on_start(self):
        """Called when user starts"""
        self.client.post("/api/auth", json={
            "api_key": "test_key"
        })
        ''', language="python")
        
        st.subheader("3. Run Load Test")
        st.code("""
# Start with 10 users, ramp up to 100
locust -f locustfile.py --host=http://localhost:8000 \\
       --users 100 --spawn-rate 10 --run-time 5m
        """, language="bash")
        
        st.subheader("4. Interpret Results")
        
        # Mock results table
        results_data = {
            "Metric": ["Total Requests", "Failures", "Median Latency", "95th Percentile", "Max Latency", "RPS"],
            "Value": ["5,432", "12 (0.2%)", "1,234 ms", "3,456 ms", "8,901 ms", "18.1"],
            "Status": ["✅ Good", "✅ Excellent", "⚠️ Acceptable", "❌ Too High", "❌ Investigate", "⚠️ Low"]
        }
        
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
        
        st.info("""
        **Key Insights from Results:**
        - 95th percentile > 3s indicates some users experiencing poor performance
        - Max latency of 9s suggests timeout issues
        - 18 RPS is low - consider horizontal scaling
        """)
        
        with st.expander("🔧 Advanced: Custom Metrics"):
            st.code('''
# Track custom metrics in Locust
from locust import events

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    if "chat" in name:
        # Log TTFT (from custom header)
        ttft = kwargs.get("response").headers.get("X-TTFT")
        if ttft:
            events.request.fire(
                request_type="TTFT",
                name="time_to_first_token",
                response_time=float(ttft),
                response_length=0
            )
            ''', language="python")

    # TAB 4: Benchmarks
    with tabs[3]:
        st.header("📊 Real-World Benchmarks")
        
        st.markdown("Compare performance across different providers and models.")
        
        # Benchmark data
        benchmark_data = {
            "Provider": ["OpenAI GPT-4 Turbo", "OpenAI GPT-3.5 Turbo", "Anthropic Claude 3 Sonnet", 
                        "Groq Llama 3 70B", "Groq Mixtral 8x7B", "Together AI Llama 3 70B"],
            "TTFT (ms)": [450, 250, 380, 120, 90, 200],
            "Tokens/sec": [28, 85, 45, 280, 520, 95],
            "Cost ($/1M tokens)": [10.0, 0.5, 3.0, 0.7, 0.24, 0.9],
            "Quality": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐"]
        }
        
        df = pd.DataFrame(benchmark_data)
        
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Tokens/sec"],
            y=df["Cost ($/1M tokens)"],
            mode='markers+text',
            marker=dict(size=df["TTFT (ms)"] / 10, color=df["TTFT (ms)"], 
                       colorscale='RdYlGn_r', showscale=True,
                       colorbar=dict(title="TTFT (ms)")),
            text=df["Provider"],
            textposition="top center"
        ))
        
        fig.update_layout(
            title="Speed vs Cost (Bubble size = TTFT)",
            xaxis_title="Tokens/Second (Higher is Better)",
            yaxis_title="Cost per 1M Tokens (Lower is Better)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **Key Takeaways:**
        - **Groq** offers 10x faster inference but limited model selection
        - **GPT-3.5 Turbo** is the best balance of speed/cost/quality
        - **Claude 3 Sonnet** excels at quality but slower than GPT-3.5
        """)

    # TAB 5: Optimization
    with tabs[4]:
        st.header("⚡ Optimization Strategies")
        
        strategies = [
            {
                "title": "1. 🌊 Always Stream Responses",
                "impact": "Reduces perceived latency by 80%",
                "code": '''
# FastAPI streaming example
from fastapi.responses import StreamingResponse

async def stream_chat(message: str):
    async for chunk in llm.astream(message):
        yield f"data: {chunk}\\n\\n"

@app.post("/chat")
async def chat(message: str):
    return StreamingResponse(
        stream_chat(message),
        media_type="text/event-stream"
    )
                '''
            },
            {
                "title": "2. 🎯 Use Smaller Models",
                "impact": "3-10x faster inference",
                "code": '''
# Model selection based on task complexity
def select_model(task_complexity):
    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # Fast, cheap
    elif task_complexity == "medium":
        return "claude-3-haiku"  # Balanced
    else:
        return "gpt-4-turbo"  # Slow but smart
                '''
            },
            {
                "title": "3. 🚀 Speculative Decoding",
                "impact": "2-3x faster for long outputs",
                "code": '''
# Use small model to draft, large model to verify
draft = small_model.generate(prompt, max_tokens=100)
final = large_model.verify_and_extend(draft)
                '''
            },
            {
                "title": "4. 💾 Caching",
                "impact": "100x faster for repeated queries",
                "code": '''
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm_call(prompt_hash):
    return llm.generate(prompt)

def smart_generate(prompt):
    h = hashlib.md5(prompt.encode()).hexdigest()
    return cached_llm_call(h)
                '''
            },
            {
                "title": "5. ⚙️ Batch Processing",
                "impact": "5x higher throughput",
                "code": '''
# Process multiple requests together
async def batch_generate(prompts: list[str]):
    # Wait for small batch or timeout
    batch = await collect_batch(prompts, max_size=10, timeout=0.1)
    
    # Single API call for all
    results = await llm.batch_generate(batch)
    
    return results
                '''
            }
        ]
        
        for strategy in strategies:
            with st.expander(strategy["title"]):
                st.success(f"**Impact**: {strategy['impact']}")
                st.code(strategy["code"], language="python")

    # TAB 6: Monitoring
    with tabs[5]:
        st.header("🔔 Monitoring & Alerting")
        
        st.markdown("""
        **You can't improve what you don't measure.** Set up monitoring from day one.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Metrics to Track")
            st.markdown("""
            **Latency Metrics:**
            - P50, P95, P99 latency
            - TTFT distribution
            - Error rate by endpoint
            
            **Business Metrics:**
            - API calls per user
            - Cost per request
            - User satisfaction (thumbs up/down)
            
            **Infrastructure:**
            - CPU/GPU utilization
            - Memory usage
            - Queue depth
            """)
        
        with col2:
            st.subheader("Alerting Thresholds")
            st.code("""
# Example alert rules
alerts:
  - name: high_latency
    condition: p95_latency > 5000ms
    action: page_oncall
    
  - name: error_spike
    condition: error_rate > 1%
    action: slack_alert
    
  - name: cost_anomaly
    condition: hourly_cost > $100
    action: email_team
            """, language="yaml")
        
        st.subheader("Recommended Tools")
        
        tools = {
            "LangSmith": "LLM-specific observability (traces, costs, feedback)",
            "Datadog": "Full-stack monitoring with APM",
            "Prometheus + Grafana": "Open-source metrics & dashboards",
            "Sentry": "Error tracking and performance monitoring"
        }
        
        for tool, desc in tools.items():
            st.markdown(f"**{tool}**: {desc}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔗 Resources
    - [LLMPerf Leaderboard](https://github.com/ray-project/llmperf-leaderboard)
    - [Artificial Analysis Benchmarks](https://artificialanalysis.ai/)
    - [Locust Documentation](https://docs.locust.io/)
    - [VLLM Inference Engine](https://github.com/vllm-project/vllm)
    """)
