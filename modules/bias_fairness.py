import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("⚖️ Bias & Fairness in AI Systems")
    
    st.markdown("""
    **"With great power comes great responsibility."** AI systems can perpetuate and amplify societal biases.
    Learn how to detect, measure, and mitigate bias in your models.
    """)
    
    tabs = st.tabs([
        "⚠️ Types of Bias",
        "📏 Fairness Metrics",
        "🧮 Interactive Calculator",
        "🛠️ Mitigation Strategies",
        "📊 Case Studies",
        "✅ Best Practices"
    ])
    
    # TAB 1: Types of Bias
    with tabs[0]:
        st.header("⚠️ Understanding AI Bias")
        
        st.markdown("""
        **Bias** occurs when an AI system produces systematically prejudiced results due to flawed assumptions in the ML process.
        """)
        
        bias_types = {
            "1. Historical Bias": {
                "definition": "Training data reflects historical prejudices and inequalities",
                "example": "Resume screening AI trained on past hires (mostly male) rejects female candidates",
                "real_case": "Amazon's recruiting tool (discontinued 2018)",
                "impact": "Perpetuates existing discrimination"
            },
            "2. Representation Bias": {
                "definition": "Certain groups are underrepresented in training data",
                "example": "Facial recognition trained mostly on white faces performs poorly on darker skin tones",
                "real_case": "MIT study: 34% error rate for dark-skinned women vs 0.8% for white men",
                "impact": "Unequal performance across demographics"
            },
            "3. Measurement Bias": {
                "definition": "Choosing the wrong features or labels as proxies",
                "example": "Using 'arrest rate' as proxy for 'crime rate' (arrests reflect policing bias)",
                "real_case": "COMPAS recidivism prediction",
                "impact": "Systematically wrong predictions for certain groups"
            },
            "4. Aggregation Bias": {
                "definition": "One-size-fits-all model ignores subgroup differences",
                "example": "Diabetes risk model trained on average population misses ethnic variations",
                "real_case": "Pulse oximeters less accurate for Black patients",
                "impact": "Medical misdiagnosis, unequal treatment"
            },
            "5. Evaluation Bias": {
                "definition": "Test data doesn't represent deployment population",
                "example": "Speech recognition tested only on American English, deployed globally",
                "real_case": "Voice assistants struggle with accents",
                "impact": "Overestimated performance, real-world failures"
            },
            "6. Deployment Bias": {
                "definition": "System used in inappropriate contexts or by wrong users",
                "example": "Emotion detection AI (dubious science) used in hiring decisions",
                "real_case": "HireVue video interviews",
                "impact": "Pseudoscientific discrimination"
            }
        }
        
        for bias_name, info in bias_types.items():
            with st.expander(f"**{bias_name}**"):
                st.markdown(f"**Definition**: {info['definition']}")
                st.markdown(f"**Example**: {info['example']}")
                st.warning(f"**Real Case**: {info['real_case']}")
                st.error(f"**Impact**: {info['impact']}")

    # TAB 2: Fairness Metrics
    with tabs[1]:
        st.header("📏 Measuring Fairness")
        
        st.markdown("""
        **"You can't improve what you don't measure."** These metrics quantify fairness.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Group Fairness Metrics")
            
            st.markdown("""
            **1. Demographic Parity (Statistical Parity)**
            - Positive prediction rate equal across groups
            - Formula: `P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)`
            - Example: 50% of men hired → 50% of women hired
            
            **2. Equal Opportunity**
            - True Positive Rate (TPR) equal across groups
            - Formula: `P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)`
            - Example: Qualified candidates have equal acceptance rates
            
            **3. Equalized Odds**
            - Both TPR and FPR equal across groups
            - Stricter than Equal Opportunity
            - Example: Both qualified and unqualified treated equally
            """)
        
        with col2:
            st.subheader("Individual Fairness Metrics")
            
            st.markdown("""
            **4. Fairness Through Awareness**
            - Similar individuals get similar predictions
            - Formula: `d(x₁, x₂) ≈ d(ŷ₁, ŷ₂)`
            - Example: Two similar loan applicants get similar scores
            
            **5. Counterfactual Fairness**
            - Prediction unchanged if sensitive attribute changed
            - Formula: `P(Ŷ | A=0, X) = P(Ŷ | A=1, X)`
            - Example: Same person, different race → same outcome
            
            **6. Calibration**
            - Predicted probabilities match actual outcomes per group
            - Formula: `P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1)`
            - Example: 70% risk score means 70% actual risk for all groups
            """)
        
        st.warning("""
        **⚠️ Impossibility Theorem**: You cannot satisfy all fairness metrics simultaneously 
        (except in trivial cases). You must choose which fairness definition matters for your use case.
        """)

    # TAB 3: Interactive Calculator
    with tabs[2]:
        st.header("🧮 Fairness Metrics Calculator")
        
        st.markdown("Calculate fairness metrics for a binary classifier across two groups.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Group A (e.g., Male)")
            a_tp = st.number_input("True Positives (A)", 0, 1000, 80, 1)
            a_fp = st.number_input("False Positives (A)", 0, 1000, 10, 1)
            a_tn = st.number_input("True Negatives (A)", 0, 1000, 85, 1)
            a_fn = st.number_input("False Negatives (A)", 0, 1000, 25, 1)
        
        with col2:
            st.subheader("Group B (e.g., Female)")
            b_tp = st.number_input("True Positives (B)", 0, 1000, 60, 1)
            b_fp = st.number_input("False Positives (B)", 0, 1000, 15, 1)
            b_tn = st.number_input("True Negatives (B)", 0, 1000, 80, 1)
            b_fn = st.number_input("False Negatives (B)", 0, 1000, 45, 1)
        
        # Calculate metrics
        def calc_metrics(tp, fp, tn, fn):
            total = tp + fp + tn + fn
            positive_rate = (tp + fp) / total if total > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            accuracy = (tp + tn) / total if total > 0 else 0
            return positive_rate, tpr, fpr, precision, accuracy
        
        a_pos_rate, a_tpr, a_fpr, a_prec, a_acc = calc_metrics(a_tp, a_fp, a_tn, a_fn)
        b_pos_rate, b_tpr, b_fpr, b_prec, b_acc = calc_metrics(b_tp, b_fp, b_tn, b_fn)
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Metrics comparison
        metrics_df = pd.DataFrame({
            "Metric": ["Positive Rate", "True Positive Rate (TPR)", "False Positive Rate (FPR)", "Precision", "Accuracy"],
            "Group A": [f"{a_pos_rate:.1%}", f"{a_tpr:.1%}", f"{a_fpr:.1%}", f"{a_prec:.1%}", f"{a_acc:.1%}"],
            "Group B": [f"{b_pos_rate:.1%}", f"{b_tpr:.1%}", f"{b_fpr:.1%}", f"{b_prec:.1%}", f"{b_acc:.1%}"],
            "Difference": [
                f"{abs(a_pos_rate - b_pos_rate):.1%}",
                f"{abs(a_tpr - b_tpr):.1%}",
                f"{abs(a_fpr - b_fpr):.1%}",
                f"{abs(a_prec - b_prec):.1%}",
                f"{abs(a_acc - b_acc):.1%}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Fairness assessment
        st.subheader("⚖️ Fairness Assessment")
        
        demo_parity_diff = abs(a_pos_rate - b_pos_rate)
        equal_opp_diff = abs(a_tpr - b_tpr)
        equal_odds_diff = abs(a_tpr - b_tpr) + abs(a_fpr - b_fpr)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Demographic Parity", 
                     "✅ Fair" if demo_parity_diff < 0.05 else "❌ Biased",
                     delta=f"{demo_parity_diff:.1%} difference")
        
        with col2:
            st.metric("Equal Opportunity",
                     "✅ Fair" if equal_opp_diff < 0.05 else "❌ Biased",
                     delta=f"{equal_opp_diff:.1%} TPR difference")
        
        with col3:
            st.metric("Equalized Odds",
                     "✅ Fair" if equal_odds_diff < 0.10 else "❌ Biased",
                     delta=f"{equal_odds_diff:.1%} total difference")

    # TAB 4: Mitigation Strategies
    with tabs[3]:
        st.header("🛠️ Bias Mitigation Strategies")
        
        strategies = {
            "Pre-Processing (Fix Data)": {
                "techniques": [
                    "Reweighting: Assign higher weights to underrepresented groups",
                    "Resampling: Oversample minority class, undersample majority",
                    "Data Augmentation: Generate synthetic examples for rare groups",
                    "Disparate Impact Remover: Transform features to remove correlation with sensitive attributes"
                ],
                "pros": "Model-agnostic, preserves model performance",
                "cons": "May lose information, doesn't guarantee fairness",
                "tools": "AIF360, Fairlearn"
            },
            "In-Processing (Fair Training)": {
                "techniques": [
                    "Adversarial Debiasing: Train model to predict Y while adversary can't predict A",
                    "Prejudice Remover: Add regularization term that penalizes discrimination",
                    "Fair Constraints: Add fairness constraints to optimization (e.g., equal TPR)",
                    "Meta Fair Classifier: Learn fair classifier from scratch"
                ],
                "pros": "Directly optimizes for fairness",
                "cons": "Requires custom training, may reduce accuracy",
                "tools": "Fairlearn, AIF360, TensorFlow Fairness Indicators"
            },
            "Post-Processing (Fix Predictions)": {
                "techniques": [
                    "Threshold Optimization: Use different decision thresholds per group",
                    "Calibrated Equalized Odds: Adjust predictions to satisfy equalized odds",
                    "Reject Option Classification: Withhold predictions near decision boundary",
                    "Equalized Odds Post-Processing: Flip predictions to achieve fairness"
                ],
                "pros": "Works with any model, easy to implement",
                "cons": "Doesn't fix root cause, may reduce overall accuracy",
                "tools": "Fairlearn, scikit-learn"
            }
        }
        
        for strategy, info in strategies.items():
            with st.expander(f"**{strategy}**"):
                st.markdown("**Techniques:**")
                for tech in info['techniques']:
                    st.markdown(f"- {tech}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Pros**: {info['pros']}")
                with col2:
                    st.warning(f"**Cons**: {info['cons']}")
                
                st.info(f"**Tools**: {info['tools']}")
        
        st.markdown("---")
        st.subheader("Code Example: Fairlearn")
        
        st.code("""
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# Base model
base_model = LogisticRegression()

# Mitigator with fairness constraint
mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=DemographicParity()  # or EqualizedOdds()
)

# Train fair model
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

# Predict
y_pred = mitigator.predict(X_test)

# Evaluate fairness
from fairlearn.metrics import MetricFrame, selection_rate

metrics = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test
)

print(metrics.by_group)
        """, language="python")

    # TAB 5: Case Studies
    with tabs[4]:
        st.header("📊 Real-World Bias Case Studies")
        
        cases = [
            {
                "title": "🏢 Amazon Recruiting Tool (2018)",
                "problem": "AI penalized resumes containing 'women's' (e.g., 'women's chess club')",
                "cause": "Trained on 10 years of male-dominated resumes",
                "outcome": "Project discontinued",
                "lesson": "Historical bias in training data perpetuates discrimination"
            },
            {
                "title": "👮 COMPAS Recidivism (ProPublica 2016)",
                "problem": "Black defendants labeled 'high risk' at 2x rate of white defendants with same history",
                "cause": "Proxy features correlated with race (zip code, social network)",
                "outcome": "Ongoing legal challenges",
                "lesson": "Seemingly neutral features can encode bias"
            },
            {
                "title": "👤 Facial Recognition (MIT 2018)",
                "problem": "34% error rate for dark-skinned women vs 0.8% for white men",
                "cause": "Training datasets (90% lighter-skinned, 75% male)",
                "outcome": "IBM, Microsoft, Amazon paused police sales",
                "lesson": "Representation bias leads to unequal performance"
            },
            {
                "title": "🏥 Healthcare Algorithm (Science 2019)",
                "problem": "Black patients assigned lower risk scores despite being sicker",
                "cause": "Used 'healthcare cost' as proxy for 'health need' (systemic access disparities)",
                "outcome": "Algorithm revised, affected millions",
                "lesson": "Measurement bias from poor proxy variables"
            },
            {
                "title": "💰 Apple Card (2019)",
                "problem": "Women offered lower credit limits than men with same financials",
                "cause": "Opaque algorithm, possible historical bias",
                "outcome": "NY investigation, algorithm audit",
                "lesson": "Lack of transparency enables undetected bias"
            }
        ]
        
        for case in cases:
            with st.expander(case["title"]):
                st.markdown(f"**Problem**: {case['problem']}")
                st.markdown(f"**Root Cause**: {case['cause']}")
                st.markdown(f"**Outcome**: {case['outcome']}")
                st.success(f"**Lesson**: {case['lesson']}")

    # TAB 6: Best Practices
    with tabs[5]:
        st.header("✅ Fairness Best Practices")
        
        st.subheader("🔍 Before Training")
        st.markdown("""
        1. **Audit Training Data**
           - Check demographic distributions
           - Identify underrepresented groups
           - Look for historical biases
        
        2. **Define Fairness Goals**
           - Which fairness metric matters for your use case?
           - Document trade-offs (accuracy vs fairness)
           - Get stakeholder buy-in
        
        3. **Identify Sensitive Attributes**
           - Race, gender, age, disability, etc.
           - Consider intersectionality (Black + Female)
           - Check for proxy features (zip code → race)
        """)
        
        st.subheader("⚙️ During Training")
        st.markdown("""
        4. **Use Fairness-Aware Algorithms**
           - Fairlearn, AIF360, TensorFlow Fairness
           - Add fairness constraints
           - Monitor fairness metrics during training
        
        5. **Validate on Diverse Test Sets**
           - Ensure test data represents all groups
           - Check performance per demographic
           - Test on edge cases
        """)
        
        st.subheader("🚀 After Deployment")
        st.markdown("""
        6. **Continuous Monitoring**
           - Track fairness metrics in production
           - Set up alerts for bias drift
           - Collect feedback from affected groups
        
        7. **Transparency & Explainability**
           - Document model limitations
           - Provide explanations for decisions
           - Enable appeals process
        
        8. **Regular Audits**
           - Third-party fairness audits
           - Red team testing
           - Update models as society evolves
        """)
        
        st.markdown("---")
        st.warning("""
        **⚠️ Important**: Fairness is not a one-time checkbox. It requires ongoing vigilance, 
        stakeholder engagement, and willingness to prioritize equity over raw performance.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔗 Resources
    - [Fairlearn Documentation](https://fairlearn.org/)
    - [AI Fairness 360 (IBM)](https://aif360.mybluemix.net/)
    - [Google's PAIR Guidebook](https://pair.withgoogle.com/guidebook/)
    - [Microsoft's Responsible AI](https://www.microsoft.com/en-us/ai/responsible-ai)
    - [ProPublica COMPAS Analysis](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
    """)
