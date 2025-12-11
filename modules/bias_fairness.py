import streamlit as st

def show():
    st.title("⚖️ Bias & Fairness in AI")
    
    st.markdown("""
    ### Implementing Ethical AI
    
    AI models magnify the biases present in their training data. 
    Reviewing for fairness is a critical QA step.
    """)
    
    tabs = st.tabs([
        "⚠️ Types of Bias",
        "📏 Metrics",
        "🛠️ Mitigation Tools"
    ])
    
    # TAB 1: Types
    with tabs[0]:
        st.header("⚠️ Common Biases")
        
        biases = [
            ("Historical Bias", "Data reflects historical prejudices (e.g., Doctors = Male, Nurses = Female)."),
            ("Representation Bias", "Certain groups are underrepresented in the training data."),
            ("Measurement Bias", "Choosing the wrong features/labels (e.g., using 'arrests' as a proxy for 'crime')."),
            ("Aggregation Bias", "One model for all groups, ignoring subgroup differences."),
        ]
        
        for name, desc in biases:
            st.markdown(f"**{name}:** {desc}")
            
        st.error("Example: An HR Application that rejects resumes from women because it trained on historical hiring data.")

    # TAB 2: Metrics
    with tabs[1]:
        st.header("📏 Measuring Fairness")
        
        st.info("How do we put a number on 'Fairness'?")
        
        st.markdown("""
        **1. Demographic Parity (Statistical Parity):**
        - The acceptance rate must be equal for all groups.
        - *Example:* If 50% of men are hired, 50% of women should be hired.
        
        **2. Equal Opportunity:**
        - True Positive Rates (TPR) must be equal.
        - *Example:* If a qualified man has an 80% chance of being hired, a qualified woman should too.
        
        **3. Equalized Odds:**
        - Both TPR and False Positive Rates (FPR) must be equal.
        """)

    # TAB 3: Tools
    with tabs[2]:
        st.header("🛠️ Mitigation Tools")
        
        st.subheader("Fairlearn")
        st.markdown("Python package to assess and improve fairness.")
        st.code('''
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score

# Measure accuracy by group
metrics = MetricFrame(
    metrics=accuracy_score,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sex_column
)

print(metrics.by_group)
# Male      0.85
# Female    0.65  <-- Disparity found!
        ''', language="python")

        st.subheader("AIF360 (AI Fairness 360)")
        st.markdown("IBM's toolkit with algorithms to fix bias (Pre-processing, In-processing, Post-processing).")
