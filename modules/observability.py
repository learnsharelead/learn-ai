import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

def show():
    st.title("📊 Observability & Monitoring for AI Systems")
    
    st.markdown("""
    **"You can't improve what you don't measure."** Production AI systems need observability 
    to debug failures, optimize costs, and ensure quality.
    """)
    
    tabs = st.tabs([
        "🔭 Tracing Fundamentals",
        "📈 Key Metrics",
        "🛠️ Tool Comparison",
        "💻 LangSmith Setup",
        "📊 Live Dashboard Demo",
        "⚠️ Alerting"
    ])
    
    # TAB 1: Tracing Fundamentals
    with tabs[0]:
        st.header("🔭 What is Tracing?")
        
        st.markdown("""
        **Tracing** captures the complete execution path of an AI request, from input to output.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Without Tracing")
            st.code("""
User: "What's the weather?"
AI: "I don't know."

❌ Why did it fail?
❌ What docs were retrieved?
❌ What was the actual prompt?
❌ How much did it cost?
            """)
            st.error("**Debugging = Guesswork**")
        
        with col2:
            st.subheader("With Tracing")
            st.code("""
✅ Input: "What's the weather?"
✅ Retrieved: 0 docs (empty!)
✅ Prompt: "Answer: {context}"
✅ LLM: GPT-4 (500ms, $0.002)
✅ Output: "I don't know."

Root cause: Retriever failed!
            """)
            st.success("**Debugging = Data-Driven**")
        
        st.markdown("---")
        st.subheader("Anatomy of a Trace")
        
        st.graphviz_chart("""
        digraph Trace {
            rankdir=TB;
            node [shape=box, style=filled];
            
            Input [label="1. User Input\n'Explain RAG'", color=lightblue];
            Retriever [label="2. Vector Search\n3 docs retrieved\n120ms", color=lightyellow];
            Prompt [label="3. Prompt Assembly\n1,234 tokens", color=lightgreen];
            LLM [label="4. LLM Call\nGPT-4 Turbo\n2.3s, $0.05", color=lightpink];
            Parser [label="5. Output Parser\nJSON validated", color=lightcyan];
            Output [label="6. Final Response\n'RAG combines...'", color=lightgray];
            
            Input -> Retriever -> Prompt -> LLM -> Parser -> Output;
        }
        """)
        
        st.info("""
        **Key Benefits:**
        - **Debug failures** - See exactly where things broke
        - **Optimize costs** - Identify expensive calls
        - **Improve quality** - A/B test prompts with real data
        - **Monitor latency** - Find bottlenecks
        """)

    # TAB 2: Key Metrics
    with tabs[1]:
        st.header("📈 Essential Observability Metrics")
        
        metrics_categories = {
            "Performance Metrics": {
                "Latency (P50, P95, P99)": "Response time distribution",
                "Tokens/Second": "Generation speed",
                "TTFT (Time to First Token)": "Perceived responsiveness",
                "Error Rate": "Failed requests %"
            },
            "Cost Metrics": {
                "Cost per Request": "Average spend per API call",
                "Daily/Monthly Spend": "Budget tracking",
                "Cost by Model": "Which models are expensive?",
                "Token Usage": "Input vs output tokens"
            },
            "Quality Metrics": {
                "User Feedback": "Thumbs up/down ratio",
                "Hallucination Rate": "Factual accuracy",
                "Retrieval Precision": "Relevant docs retrieved",
                "Output Validation": "Schema compliance"
            },
            "Business Metrics": {
                "Active Users": "DAU/MAU",
                "Requests per User": "Engagement",
                "Conversation Length": "Session depth",
                "Retention Rate": "User stickiness"
            }
        }
        
        for category, metrics in metrics_categories.items():
            with st.expander(f"**{category}**"):
                for metric, description in metrics.items():
                    st.markdown(f"**{metric}**: {description}")

    # TAB 3: Tool Comparison
    with tabs[2]:
        st.header("🛠️ Observability Tools Comparison")
        
        tools_data = {
            "Tool": ["LangSmith", "Weights & Biases", "Phoenix (Arize)", "Helicone", "LangFuse"],
            "Best For": ["LangChain apps", "Experiment tracking", "Open source", "Cost tracking", "Self-hosted"],
            "Pricing": ["$39/mo", "$50/user/mo", "Free", "$20/mo", "Free"],
            "Tracing": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐"],
            "Evals": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐"],
            "Ease of Use": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐"]
        }
        
        df = pd.DataFrame(tools_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **LangSmith** (Recommended)
            - ✅ Deep LangChain integration
            - ✅ Replay traces
            - ✅ Dataset testing
            - ✅ Prompt versioning
            - ❌ Expensive at scale
            """)
        
        with col2:
            st.markdown("""
            **Phoenix** (Open Source)
            - ✅ Completely free
            - ✅ Local deployment
            - ✅ Good visualizations
            - ❌ Limited evals
            - ❌ No cloud hosting
            """)
        
        with col3:
            st.markdown("""
            **Helicone** (Cost Focus)
            - ✅ Cheapest option
            - ✅ Great cost analytics
            - ✅ Caching layer
            - ❌ Basic tracing
            - ❌ Limited features
            """)

    # TAB 4: LangSmith Setup
    with tabs[3]:
        st.header("💻 LangSmith Setup (5 Minutes)")
        
        st.markdown("**LangSmith** is the industry standard for LangChain observability.")
        
        st.subheader("Step 1: Get API Key")
        st.markdown("""
        1. Go to [smith.langchain.com](https://smith.langchain.com)
        2. Sign up (free tier: 5K traces/month)
        3. Create API key
        """)
        
        st.subheader("Step 2: Configure Environment")
        st.code("""
# .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=my-ai-app
        """, language="bash")
        
        st.subheader("Step 3: Install & Run")
        st.code("""
pip install langsmith langchain
        """, language="bash")
        
        st.code("""
# Your existing code - no changes needed!
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

chain = prompt | llm

# This automatically logs to LangSmith
result = chain.invoke({"topic": "AI"})
        """, language="python")
        
        st.success("✅ That's it! Check smith.langchain.com for traces")
        
        st.markdown("---")
        st.subheader("Advanced: Custom Metadata")
        st.code("""
from langsmith import traceable

@traceable(
    run_type="llm",
    name="custom_chain",
    metadata={"version": "v2", "user_id": "123"}
)
def my_chain(input_text):
    # Your logic here
    return llm.invoke(input_text)
        """, language="python")

    # TAB 5: Live Dashboard Demo
    with tabs[4]:
        st.header("📊 Simulated Observability Dashboard")
        
        st.markdown("**What a production dashboard looks like:**")
        
        # Generate mock data
        hours = list(range(24))
        requests = [random.randint(100, 500) for _ in hours]
        latency_p50 = [random.randint(800, 1500) for _ in hours]
        latency_p95 = [random.randint(2000, 4000) for _ in hours]
        error_rate = [random.uniform(0, 2) for _ in hours]
        cost = [random.uniform(5, 20) for _ in hours]
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests (24h)", f"{sum(requests):,}", delta="+12%")
        with col2:
            avg_latency = sum(latency_p50) / len(latency_p50)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms", delta="-50ms")
        with col3:
            avg_error = sum(error_rate) / len(error_rate)
            st.metric("Error Rate", f"{avg_error:.2f}%", delta="-0.3%")
        with col4:
            total_cost = sum(cost)
            st.metric("Cost (24h)", f"${total_cost:.2f}", delta="+$2.50")
        
        # Requests over time
        fig1 = px.line(
            x=hours,
            y=requests,
            labels={'x': 'Hour of Day', 'y': 'Requests'},
            title='Requests per Hour'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Latency distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hours, y=latency_p50, name='P50', mode='lines'))
        fig2.add_trace(go.Scatter(x=hours, y=latency_p95, name='P95', mode='lines'))
        fig2.update_layout(
            title='Latency Distribution',
            xaxis_title='Hour of Day',
            yaxis_title='Latency (ms)'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Model usage breakdown
        model_data = {
            "Model": ["GPT-4 Turbo", "GPT-3.5 Turbo", "Claude 3 Sonnet", "Llama 3 70B"],
            "Requests": [1200, 3500, 800, 500],
            "Avg Cost": [0.05, 0.002, 0.015, 0.001],
            "Total Cost": [60, 7, 12, 0.5]
        }
        
        st.subheader("Model Usage Breakdown")
        st.dataframe(pd.DataFrame(model_data), use_container_width=True)

    # TAB 6: Alerting
    with tabs[5]:
        st.header("⚠️ Alerting & Incident Response")
        
        st.markdown("""
        **Proactive monitoring** catches issues before users complain.
        """)
        
        st.subheader("Common Alert Rules")
        
        alerts = [
            {
                "name": "🔴 High Error Rate",
                "condition": "error_rate > 5% for 5 minutes",
                "action": "Page on-call engineer",
                "severity": "Critical"
            },
            {
                "name": "🟡 Latency Spike",
                "condition": "p95_latency > 5000ms for 10 minutes",
                "action": "Slack alert to #ai-ops",
                "severity": "Warning"
            },
            {
                "name": "🟠 Cost Anomaly",
                "condition": "hourly_cost > $100 (2x baseline)",
                "action": "Email to finance team",
                "severity": "High"
            },
            {
                "name": "🔵 Low Quality Score",
                "condition": "thumbs_down_ratio > 30%",
                "action": "Create Jira ticket",
                "severity": "Medium"
            }
        ]
        
        for alert in alerts:
            with st.expander(f"{alert['name']} - {alert['severity']}"):
                st.markdown(f"**Condition**: `{alert['condition']}`")
                st.markdown(f"**Action**: {alert['action']}")
        
        st.markdown("---")
        st.subheader("Example Alert Configuration")
        
        st.code("""
# Prometheus AlertManager config
groups:
  - name: ai_system_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 5%"
          description: "{{ $value }}% of requests failing"
          
      - alert: HighCost
        expr: sum(rate(llm_cost_dollars[1h])) > 100
        for: 10m
        labels:
          severity: high
        annotations:
          summary: "Hourly cost exceeds $100"
        """, language="yaml")
        
        st.markdown("---")
        st.subheader("Incident Response Checklist")
        
        st.markdown("""
        **When an alert fires:**
        
        1. ✅ **Acknowledge** - Silence alert, claim ownership
        2. ✅ **Assess** - Check dashboard, review traces
        3. ✅ **Mitigate** - Rollback, scale up, or disable feature
        4. ✅ **Communicate** - Update status page, notify stakeholders
        5. ✅ **Root Cause** - Post-mortem analysis
        6. ✅ **Prevent** - Add tests, improve monitoring
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔗 Resources
    - [LangSmith Documentation](https://docs.smith.langchain.com/)
    - [Phoenix (Arize) GitHub](https://github.com/Arize-ai/phoenix)
    - [Weights & Biases LLM Guide](https://wandb.ai/site/solutions/llmops)
    - [Helicone](https://www.helicone.ai/)
    - [LangFuse (Open Source)](https://langfuse.com/)
    """)
