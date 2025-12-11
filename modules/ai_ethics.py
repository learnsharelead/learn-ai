import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("⚖️ AI Ethics, Bias & Responsible AI")
    
    st.markdown("""
    AI is powerful, but with great power comes great responsibility. 
    This module covers the ethical considerations every AI practitioner MUST understand.
    """)
    
    tabs = st.tabs([
        "🎭 Bias in AI",
        "⚠️ Case Studies",
        "✅ Fairness Metrics",
        "🔐 Privacy & Security",
        "📜 Regulations"
    ])
    
    # TAB 1: Bias in AI
    with tabs[0]:
        st.header("🎭 What is AI Bias?")
        
        st.markdown("""
        **AI Bias** occurs when a model produces systematically unfair outcomes for certain groups.
        
        > "AI doesn't create bias. It amplifies existing biases in data."
        """)
        
        st.subheader("Sources of Bias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Bias")
            st.info("""
            - **Sampling Bias:** Training data doesn't represent the real world
            - **Historical Bias:** Data reflects past discrimination
            - **Label Bias:** Human labelers have unconscious biases
            """)
            
        with col2:
            st.markdown("### 🤖 Model Bias")
            st.info("""
            - **Algorithmic Bias:** Model design favors certain outcomes
            - **Aggregation Bias:** One model for diverse groups
            - **Evaluation Bias:** Metrics don't capture fairness
            """)
        
        st.subheader("Example: Facial Recognition Bias")
        
        # Mock data showing disparity
        groups = ["Lighter-Skinned Males", "Lighter-Skinned Females", "Darker-Skinned Males", "Darker-Skinned Females"]
        error_rates = [0.8, 3.5, 12.0, 34.7]  # Based on real studies
        
        fig = go.Figure(go.Bar(x=groups, y=error_rates, marker_color=['green', 'yellow', 'orange', 'red']))
        fig.update_layout(title="Facial Recognition Error Rates by Demographic (MIT Study)", yaxis_title="Error Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("⚠️ Error rates can be **43x higher** for darker-skinned females compared to lighter-skinned males!")
    
    # TAB 2: Case Studies
    with tabs[1]:
        st.header("⚠️ Real-World AI Failures")
        
        cases = [
            {
                "title": "Amazon's Resume Screener (2018)",
                "problem": "AI learned to downgrade resumes with words like 'women's' (e.g., 'women's chess club').",
                "cause": "Trained on historical hiring data from a male-dominated industry.",
                "lesson": "Historical data encodes historical biases."
            },
            {
                "title": "COMPAS Recidivism Algorithm",
                "problem": "Black defendants were flagged as 'high risk' at nearly twice the rate of white defendants.",
                "cause": "Proxy variables correlated with race (zip code, employment).",
                "lesson": "Even 'race-blind' models can be discriminatory."
            },
            {
                "title": "Healthcare Algorithm (2019)",
                "problem": "Algorithm gave Black patients lower risk scores, reducing their access to care.",
                "cause": "Used healthcare spending as a proxy for health need. Spending ≠ Need.",
                "lesson": "Proxy variables can encode systemic inequalities."
            },
            {
                "title": "Google Photos Gorilla Incident (2015)",
                "problem": "Auto-tagging labeled Black people as 'Gorillas'.",
                "cause": "Training data lacked diversity. Model couldn't generalize.",
                "lesson": "Representation in training data matters."
            }
        ]
        
        for case in cases:
            with st.expander(f"📌 {case['title']}"):
                st.error(f"**Problem:** {case['problem']}")
                st.warning(f"**Root Cause:** {case['cause']}")
                st.success(f"**Lesson:** {case['lesson']}")
    
    # TAB 3: Fairness Metrics
    with tabs[2]:
        st.header("✅ Measuring Fairness")
        
        st.markdown("""
        There is no single definition of "fairness". Different metrics capture different notions.
        """)
        
        st.subheader("Common Fairness Metrics")
        
        st.markdown("""
        | Metric | Definition | Use Case |
        |--------|------------|----------|
        | **Demographic Parity** | Equal positive prediction rates across groups | Hiring, Lending |
        | **Equalized Odds** | Equal TPR and FPR across groups | Criminal Justice |
        | **Predictive Parity** | Equal precision across groups | Medical Diagnosis |
        | **Calibration** | Predicted probabilities match actual outcomes per group | Risk Assessment |
        """)
        
        st.warning("⚠️ **Impossibility Theorem:** You often can't satisfy all fairness metrics simultaneously!")
        
        st.subheader("Interactive: Demographic Parity Check")
        
        col1, col2 = st.columns(2)
        with col1:
            group_a_approved = st.slider("Group A Approval Rate (%)", 0, 100, 70)
        with col2:
            group_b_approved = st.slider("Group B Approval Rate (%)", 0, 100, 50)
        
        disparity = abs(group_a_approved - group_b_approved)
        
        fig = go.Figure(go.Bar(x=["Group A", "Group B"], y=[group_a_approved, group_b_approved], marker_color=['blue', 'orange']))
        fig.update_layout(title="Approval Rates by Group", yaxis_title="Approval Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        if disparity < 10:
            st.success(f"✅ Disparity: {disparity}% - Reasonably fair (80% rule threshold)")
        else:
            st.error(f"❌ Disparity: {disparity}% - May indicate discrimination")
    
    # TAB 4: Privacy & Security
    with tabs[3]:
        st.header("🔐 Privacy & Security in AI")
        
        st.subheader("Key Concerns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🕵️ Data Privacy")
            st.markdown("""
            - **Data Minimization:** Only collect what you need
            - **Anonymization:** Remove PII before training
            - **Differential Privacy:** Add noise to protect individuals
            - **Federated Learning:** Train on decentralized data
            """)
            
        with col2:
            st.markdown("### 🛡️ Model Security")
            st.markdown("""
            - **Adversarial Attacks:** Fooling models with small perturbations
            - **Model Inversion:** Extracting training data from models
            - **Prompt Injection:** Manipulating LLMs with malicious inputs
            - **Data Poisoning:** Corrupting training data
            """)
        
        st.subheader("Example: Adversarial Attack")
        st.markdown("""
        A tiny, invisible change to an image can completely fool a classifier:
        
        | Original Image | Perturbation | Adversarial Image |
        |---|---|---|
        | 🐼 Panda (99.9%) | + (invisible noise) = | 🦧 Gibbon (99.3%) |
        """)
        st.caption("The perturbation is imperceptible to humans but completely fools the AI.")
    
    # TAB 5: Regulations
    with tabs[4]:
        st.header("📜 AI Regulations & Guidelines")
        
        st.subheader("Key Frameworks")
        
        st.markdown("""
        | Region | Regulation | Key Points |
        |--------|------------|------------|
        | 🇪🇺 EU | **EU AI Act (2024)** | Risk-based classification. Bans social scoring. Requires transparency. |
        | 🇺🇸 USA | **AI Bill of Rights (2022)** | Non-binding principles. Safe AI, Algorithmic discrimination protections. |
        | 🇮🇳 India | **DPDP Act (2023)** | Data protection. Consent requirements. |
        | 🌍 Global | **OECD AI Principles** | Transparency, Accountability, Human-centered values. |
        """)
        
        st.subheader("Responsible AI Checklist")
        
        checklist = [
            "✅ Is the training data representative and unbiased?",
            "✅ Have we tested for disparate impact across groups?",
            "✅ Can we explain the model's decisions?",
            "✅ Is there a human-in-the-loop for high-stakes decisions?",
            "✅ Do users consent to data collection?",
            "✅ Is there a process for redress if the AI harms someone?",
            "✅ Are we transparent about AI limitations?"
        ]
        
        for item in checklist:
            st.markdown(item)
        
        st.info("💡 **Pro Tip:** Ethics is not a one-time checkbox. It's an ongoing process!")
