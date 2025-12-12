import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

def show():
    st.title("📈 A/B Testing for AI Systems")
    
    st.markdown("""
    **A/B Testing** is the gold standard for validating improvements in AI systems.
    Whether you're testing prompts, models, or UI changes, statistical rigor prevents costly mistakes.
    """)
    
    tabs = st.tabs([
        "🧪 Fundamentals",
        "🔢 Significance Calculator",
        "📏 Sample Size Estimator",
        "📊 Real Examples",
        "⚠️ Common Pitfalls"
    ])
    
    # TAB 1: Fundamentals
    with tabs[0]:
        st.header("🧪 A/B Testing Fundamentals")
        
        st.markdown("""
        ### The Scientific Method for AI
        
        **Scenario**: You think adding "think step by step" to your prompt improves accuracy.
        **Question**: How do you prove it?
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **❌ Bad Approach:**
            - Test on 5 examples
            - "It looks better!"
            - Deploy to production
            
            **Result**: Confirmation bias, no real improvement
            """)
        
        with col2:
            st.markdown("""
            **✅ Good Approach:**
            - Define hypothesis
            - Test on 100+ examples
            - Calculate statistical significance
            - Only deploy if p < 0.05
            
            **Result**: Confident, data-driven decision
            """)
        
        st.subheader("The A/B Testing Process")
        
        st.graphviz_chart("""
        digraph AB {
            rankdir=TB;
            node [shape=box, style=filled];
            
            H [label="1. Hypothesis\n'Variant B is better'", color=lightblue];
            D [label="2. Design Experiment\nDefine metrics, sample size", color=lightyellow];
            R [label="3. Run Test\nRandomly assign users", color=lightgreen];
            A [label="4. Analyze Results\nCalculate significance", color=lightpink];
            Dec [label="5. Decision", color=lightcyan];
            
            H -> D -> R -> A -> Dec;
        }
        """)
        
        st.markdown("""
        ### Key Concepts
        
        **Null Hypothesis (H₀)**: There is no difference between A and B  
        **Alternative Hypothesis (H₁)**: B is better than A  
        **p-value**: Probability that the observed difference is due to chance  
        **Significance Level (α)**: Threshold for rejecting H₀ (usually 0.05)
        
        **Decision Rule**: If p < 0.05, reject H₀ and conclude B is better
        """)

    # TAB 2: Significance Calculator
    with tabs[1]:
        st.header("🔢 Statistical Significance Calculator")
        
        st.markdown("Test if the difference between two variants is statistically significant.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variant A (Control)")
            n_a = st.number_input("Sample Size A:", min_value=10, value=100, step=10)
            success_a = st.number_input("Successes A:", min_value=0, max_value=n_a, value=75)
            
        with col2:
            st.subheader("Variant B (Treatment)")
            n_b = st.number_input("Sample Size B:", min_value=10, value=100, step=10)
            success_b = st.number_input("Successes B:", min_value=0, max_value=n_b, value=85)
        
        # Calculate metrics
        rate_a = success_a / n_a
        rate_b = success_b / n_b
        lift = ((rate_b - rate_a) / rate_a) * 100 if rate_a > 0 else 0
        
        # Chi-squared test
        contingency_table = np.array([
            [success_a, n_a - success_a],
            [success_b, n_b - success_b]
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Variant A Rate", f"{rate_a:.1%}")
        with col2:
            st.metric("Variant B Rate", f"{rate_b:.1%}", delta=f"{lift:+.1f}%")
        with col3:
            st.metric("p-value", f"{p_value:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            st.success(f"""
            ✅ **Statistically Significant!**
            
            With 95% confidence, Variant B is {abs(lift):.1f}% {'better' if lift > 0 else 'worse'} than Variant A.
            
            **Recommendation**: {'Deploy Variant B' if lift > 0 else 'Stick with Variant A'}
            """)
        else:
            st.warning(f"""
            ⚠️ **Not Statistically Significant**
            
            The {abs(lift):.1f}% difference could be due to random chance (p = {p_value:.4f}).
            
            **Recommendation**: Need more data or the difference is too small to matter.
            """)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Variant A',
            x=['Success Rate'],
            y=[rate_a],
            marker_color='lightblue',
            text=[f"{rate_a:.1%}"],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Variant B',
            x=['Success Rate'],
            y=[rate_b],
            marker_color='lightgreen',
            text=[f"{rate_b:.1%}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Conversion Rate Comparison",
            yaxis_title="Success Rate",
            yaxis=dict(range=[0, 1]),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Sample Size Estimator
    with tabs[2]:
        st.header("📏 Sample Size Estimator")
        
        st.markdown("""
        **Question**: How many samples do I need to detect a meaningful difference?
        
        **Answer**: It depends on:
        - Baseline conversion rate
        - Minimum detectable effect (MDE)
        - Desired statistical power
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline = st.slider("Baseline Conversion Rate:", 0.01, 0.50, 0.10, 0.01, format="%.2f")
            mde = st.slider("Minimum Detectable Effect (%):", 1, 50, 10, 1)
        
        with col2:
            alpha = st.select_slider("Significance Level (α):", options=[0.01, 0.05, 0.10], value=0.05)
            power = st.select_slider("Statistical Power (1-β):", options=[0.70, 0.80, 0.90], value=0.80)
        
        # Calculate sample size (simplified formula)
        # Using normal approximation for proportions
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        p1 = baseline
        p2 = baseline * (1 + mde/100)
        p_avg = (p1 + p2) / 2
        
        n_per_variant = ((z_alpha + z_beta)**2 * 2 * p_avg * (1 - p_avg)) / ((p2 - p1)**2)
        n_per_variant = int(np.ceil(n_per_variant))
        
        st.markdown("---")
        st.subheader("📊 Required Sample Size")
        
        st.metric("Samples per Variant", f"{n_per_variant:,}")
        st.metric("Total Samples Needed", f"{n_per_variant * 2:,}")
        
        st.info(f"""
        **Interpretation:**
        
        To detect a **{mde}% improvement** over a **{baseline:.1%} baseline** with:
        - **{int(power*100)}% power** (probability of detecting a real effect)
        - **{int((1-alpha)*100)}% confidence** (avoiding false positives)
        
        You need **{n_per_variant:,} samples per variant**.
        """)
        
        # Show how sample size changes with MDE
        mde_range = range(1, 51, 2)
        sample_sizes = []
        
        for m in mde_range:
            p2_temp = baseline * (1 + m/100)
            n_temp = ((z_alpha + z_beta)**2 * 2 * p_avg * (1 - p_avg)) / ((p2_temp - baseline)**2)
            sample_sizes.append(int(np.ceil(n_temp)))
        
        fig = px.line(
            x=list(mde_range),
            y=sample_sizes,
            labels={'x': 'Minimum Detectable Effect (%)', 'y': 'Required Sample Size'},
            title='Sample Size vs Effect Size'
        )
        fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Practical limit")
        
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Real Examples
    with tabs[3]:
        st.header("📊 Real-World A/B Test Examples")
        
        examples = [
            {
                "title": "🤖 Prompt Engineering: Chain-of-Thought",
                "context": "Testing if 'Let's think step by step' improves math accuracy",
                "variant_a": "Solve: 2x + 5 = 10",
                "variant_b": "Solve: 2x + 5 = 10. Let's think step by step.",
                "result": "Variant B: 87% accuracy vs Variant A: 72% (p < 0.001)",
                "decision": "✅ Deploy Variant B - 21% improvement, highly significant"
            },
            {
                "title": "🎨 UI Change: Button Color",
                "context": "Testing if green CTA button increases clicks vs blue",
                "variant_a": "Blue button: 'Start Free Trial'",
                "variant_b": "Green button: 'Start Free Trial'",
                "result": "Variant B: 12.3% CTR vs Variant A: 12.1% CTR (p = 0.42)",
                "decision": "❌ No significant difference - keep blue (brand consistency)"
            },
            {
                "title": "🧠 Model Selection: GPT-4 vs Claude",
                "context": "Testing which model produces better customer support responses",
                "variant_a": "GPT-4 Turbo",
                "variant_b": "Claude 3 Sonnet",
                "result": "Variant B: 4.2/5 avg rating vs Variant A: 3.9/5 (p < 0.05)",
                "decision": "✅ Switch to Claude - better user satisfaction"
            }
        ]
        
        for i, ex in enumerate(examples):
            with st.expander(f"**Example {i+1}**: {ex['title']}"):
                st.markdown(f"**Context**: {ex['context']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Variant A (Control)**  \n{ex['variant_a']}")
                with col2:
                    st.markdown(f"**Variant B (Treatment)**  \n{ex['variant_b']}")
                
                st.markdown(f"**Result**: {ex['result']}")
                st.markdown(f"**Decision**: {ex['decision']}")

    # TAB 5: Common Pitfalls
    with tabs[4]:
        st.header("⚠️ Common A/B Testing Pitfalls")
        
        pitfalls = [
            {
                "title": "🚫 Peeking at Results Early",
                "problem": "Checking p-value every 10 samples and stopping when p < 0.05",
                "why_bad": "Inflates false positive rate from 5% to 30%+",
                "solution": "Pre-define sample size and only check once at the end"
            },
            {
                "title": "🚫 Testing Too Many Variants",
                "problem": "Running A/B/C/D/E/F tests simultaneously",
                "why_bad": "Multiple comparisons problem - increases false positives",
                "solution": "Use Bonferroni correction: α_adjusted = 0.05 / num_tests"
            },
            {
                "title": "🚫 Ignoring Seasonality",
                "problem": "Running test Mon-Fri, deploying on weekend",
                "why_bad": "Weekend users behave differently",
                "solution": "Run tests for full weeks, include all user segments"
            },
            {
                "title": "🚫 Confusing Statistical vs Practical Significance",
                "problem": "0.1% improvement is significant (p < 0.05) but costs $10K/month",
                "why_bad": "Statistically real but economically worthless",
                "solution": "Always consider business impact, not just p-value"
            }
        ]
        
        for pitfall in pitfalls:
            st.subheader(pitfall['title'])
            st.markdown(f"**Problem**: {pitfall['problem']}")
            st.markdown(f"**Why It's Bad**: {pitfall['why_bad']}")
            st.success(f"**Solution**: {pitfall['solution']}")
            st.markdown("---")
    
    # Footer
    st.markdown("""
    ### 🔗 Recommended Tools
    - **Promptfoo**: A/B testing for prompts ([promptfoo.dev](https://promptfoo.dev))
    - **Statsig**: Feature flags + A/B testing platform
    - **GrowthBook**: Open-source experimentation platform
    - **Optimizely**: Enterprise A/B testing
    
    ### 📚 Further Reading
    - [Trustworthy Online Controlled Experiments](https://experimentguide.com/) (The A/B Testing Bible)
    - [Evan Miller's A/B Tools](https://www.evanmiller.org/ab-testing/)
    """)
