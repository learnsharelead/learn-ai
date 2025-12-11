import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def show():
    st.title("👨‍🏫 Module 3: Supervised Learning")
    
    st.markdown("""
    ### Learning with a Teacher - Simple Explanations + Formal Notes!
    """)
    
    tabs = st.tabs(["📚 The Concept", "📈 Regression", "✅ Classification", "🌳 Decision Trees", "📝 Complete Notes"])
    
    # TAB 1: Concept
    with tabs[0]:
        st.header("📚 What is Supervised Learning?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **Learning with flashcards!**
            
            You see questions WITH answers:
            - 🍎 [picture] → "Apple"
            - 🐕 [picture] → "Dog"
            - 📧 [spam email] → "Spam"
            
            After many examples, you learn the patterns!
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Supervised Learning** is a type of machine learning 
            where the model is trained on labeled data, meaning 
            each training example is paired with an output label. 
            The goal is to learn a mapping function from inputs 
            to outputs that can generalize to unseen data.
            
            **Formula:** Learn f(X) → Y
            """)
        
        st.markdown("---")
        st.subheader("Two Types of Problems")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📈 Regression
            
            **🍕 Simple:** Predicting a NUMBER
            - How much will this house cost?
            - What temperature tomorrow?
            
            **📘 Formal:** Regression is a supervised learning 
            task where the target variable is continuous 
            (numerical). The model learns to predict a 
            real-valued output.
            
            **Examples:** Linear Regression, Polynomial Regression
            """)
            
        with col2:
            st.markdown("""
            ### ✅ Classification
            
            **🍕 Simple:** Predicting a CATEGORY
            - Is this email spam or not?
            - What animal is in this photo?
            
            **📘 Formal:** Classification is a supervised learning 
            task where the target variable is categorical 
            (discrete). The model predicts which class/category 
            an input belongs to.
            
            **Examples:** Logistic Regression, Decision Trees
            """)
    
    # TAB 2: Regression
    with tabs[1]:
        st.header("📈 Regression (Predicting Numbers)")
        
        st.subheader("🍕 The Ice Cream Example")
        
        st.markdown("""
        You run an ice cream shop. You notice:
        - Hot day → More sales 🍦🍦🍦
        - Cold day → Less sales 🍦
        
        Can AI predict tomorrow's sales based on temperature?
        """)
        
        # Generate data
        np.random.seed(42)
        temp = np.array([60, 65, 70, 75, 80, 85, 90, 95])
        sales = temp * 5 - 100 + np.random.randn(8) * 20
        
        # Fit model
        model = LinearRegression()
        model.fit(temp.reshape(-1, 1), sales)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temp, y=sales, mode='markers', name='Actual Sales',
                                 marker=dict(size=15, color='blue')))
        
        line_x = np.linspace(55, 100, 50)
        line_y = model.predict(line_x.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='AI Prediction Line',
                                 line=dict(color='red', width=3)))
        
        fig.update_layout(title="🍦 Ice Cream Sales vs Temperature",
                         xaxis_title="Temperature (°F)", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📘 Formal Notes: Linear Regression")
        
        with st.expander("📖 Mathematical Definition"):
            st.markdown("""
            **Linear Regression** finds the best-fitting straight line 
            through data points.
            
            **Formula:**
            """)
            st.latex(r"y = mx + b")
            st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \epsilon")
            
            st.markdown("""
            Where:
            - **y** = Predicted value (dependent variable)
            - **x** = Input feature(s) (independent variables)
            - **β₀** = Intercept (y-value when x=0)
            - **β₁** = Slope/Coefficient (how much y changes per unit x)
            - **ε** = Error term
            """)
        
        with st.expander("📖 How It Learns (Least Squares)"):
            st.markdown("""
            **Objective:** Minimize the sum of squared errors
            """)
            st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
            
            st.markdown("""
            - **yᵢ** = Actual value
            - **ŷᵢ** = Predicted value
            - **n** = Number of data points
            
            The algorithm finds β values that minimize this error.
            """)
        
        st.subheader("🎮 Make a Prediction!")
        
        new_temp = st.slider("Tomorrow's Temperature (°F):", 50, 100, 85)
        predicted = model.predict([[new_temp]])[0]
        
        st.success(f"**Prediction:** At {new_temp}°F, expect ${predicted:.0f} in sales")
    
    # TAB 3: Classification
    with tabs[2]:
        st.header("✅ Classification (Predicting Categories)")
        
        st.subheader("🍕 The Spam Email Example")
        
        st.markdown("""
        Gmail needs to decide: **Spam or Not Spam?**
        
        It looks at patterns in the email content.
        """)
        
        emails = pd.DataFrame({
            'Email Subject': [
                "Meeting tomorrow at 3pm",
                "FREE iPhone!!! CLICK NOW!!!",
                "Your order has shipped",
                "CONGRATULATIONS YOU WON $1M",
            ],
            'ALL CAPS Count': [0, 3, 0, 4],
            'Exclamation Marks': [0, 4, 0, 0],
            'Label': ['✅ Normal', '🚫 Spam', '✅ Normal', '🚫 Spam']
        })
        st.dataframe(emails, use_container_width=True)
        
        st.markdown("**AI learns: More CAPS + more exclamation = probably SPAM!**")
        
        st.markdown("---")
        
        st.subheader("📘 Formal Notes: Classification")
        
        with st.expander("📖 Types of Classification"):
            st.markdown("""
            **Binary Classification:**
            - Two classes: Yes/No, Spam/Not Spam, 0/1
            - Example: Logistic Regression
            
            **Multi-class Classification:**
            - More than 2 classes: Dog/Cat/Bird
            - Example: Softmax classifier
            
            **Multi-label Classification:**
            - Multiple labels per sample: A movie can be "Action" AND "Comedy"
            """)
        
        with st.expander("📖 Logistic Regression"):
            st.markdown("""
            **Despite the name, it's for CLASSIFICATION, not regression!**
            
            **Formula (Sigmoid function):**
            """)
            st.latex(r"P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...")
            
            st.markdown("""
            - **Output:** Probability between 0 and 1
            - **Decision:** If P > 0.5, predict class 1, else class 0
            """)
        
        with st.expander("📖 Common Algorithms"):
            st.markdown("""
            | Algorithm | Pros | Cons |
            |-----------|------|------|
            | **Logistic Regression** | Fast, interpretable | Linear boundaries only |
            | **Decision Tree** | Interpretable, handles non-linear | Overfits easily |
            | **Random Forest** | Accurate, handles overfitting | Less interpretable |
            | **SVM** | Works in high dimensions | Slow on large data |
            | **Neural Networks** | Very powerful | Needs lots of data |
            """)
    
    # TAB 4: Decision Trees
    with tabs[3]:
        st.header("🌳 Decision Trees")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **Like playing 20 Questions!**
            
            - Is it raining? → Yes/No
            - Is it cold? → Yes/No
            - Go outside or stay home?
            
            A flowchart of yes/no questions!
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Decision Tree** is a supervised learning algorithm 
            that creates a tree-like model of decisions. It 
            recursively splits the data based on feature values
            to maximize information gain or minimize impurity.
            """)
        
        st.subheader("Visual Example")
        
        st.graphviz_chart("""
        digraph DecisionTree {
            node [shape=box, style=filled];
            
            Q1 [label="Is it raining?", fillcolor=lightyellow];
            Q2 [label="Is it cold?", fillcolor=lightblue];
            Q3 [label="Have umbrella?", fillcolor=lightblue];
            
            A1 [label="Stay Home", fillcolor=lightcoral, shape=ellipse];
            A2 [label="Go Outside!", fillcolor=lightgreen, shape=ellipse];
            A3 [label="Wear Jacket", fillcolor=lightgreen, shape=ellipse];
            A4 [label="Use Umbrella", fillcolor=lightgreen, shape=ellipse];
            
            Q1 -> Q2 [label="No"];
            Q1 -> Q3 [label="Yes"];
            Q2 -> A2 [label="No"];
            Q2 -> A3 [label="Yes"];
            Q3 -> A1 [label="No"];
            Q3 -> A4 [label="Yes"];
        }
        """)
        
        with st.expander("📖 How Trees Split (Gini Impurity)"):
            st.markdown("""
            **Gini Impurity:** Measures how "mixed" the classes are
            """)
            st.latex(r"Gini = 1 - \sum_{i=1}^{n} p_i^2")
            
            st.markdown("""
            - **Pure node (one class):** Gini = 0 ✅
            - **Mixed 50/50:** Gini = 0.5 ❌
            
            The algorithm finds splits that minimize Gini (purer groups).
            """)
        
        with st.expander("📖 Information Gain (Entropy)"):
            st.markdown("""
            **Entropy:** Measures impurity using information theory
            """)
            st.latex(r"Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i)")
            st.latex(r"Information\ Gain = Entropy(parent) - weighted\ Entropy(children)")
            
            st.markdown("""
            Choose the split that gives maximum information gain!
            """)
    
    # TAB 5: Complete Notes
    with tabs[4]:
        st.header("📝 Complete Notes (Copy for Reference)")
        
        st.subheader("🔑 Key Definitions")
        
        definitions = {
            "Supervised Learning": "ML approach where models learn from labeled training data (input-output pairs) to map inputs to correct outputs.",
            
            "Regression": "Predicting continuous numerical values. Output is a real number (e.g., price, temperature).",
            
            "Classification": "Predicting categorical/discrete values. Output is a class label (e.g., spam/not spam).",
            
            "Training Data": "Labeled examples used to teach the model. Contains features (X) and labels (Y).",
            
            "Test Data": "Held-out data used to evaluate model performance on unseen examples.",
            
            "Features (X)": "Input variables/attributes used for prediction. Also called independent variables.",
            
            "Labels (Y)": "Target variable the model predicts. Also called dependent variable or response.",
            
            "Overfitting": "Model performs well on training data but poorly on new data. It memorized rather than learned.",
            
            "Underfitting": "Model is too simple to capture patterns. Poor performance on both training and test data.",
            
            "Loss Function": "Measures how wrong predictions are. The goal is to minimize this during training.",
        }
        
        for term, definition in definitions.items():
            with st.expander(f"📖 {term}"):
                st.code(f"{term}: {definition}", language="text")
        
        st.subheader("📊 Key Formulas")
        
        with st.expander("📐 All Formulas"):
            st.markdown("**Linear Regression:**")
            st.latex(r"y = \beta_0 + \beta_1 x + \epsilon")
            
            st.markdown("**Mean Squared Error:**")
            st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
            
            st.markdown("**Logistic (Sigmoid) Function:**")
            st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
            
            st.markdown("**Gini Impurity:**")
            st.latex(r"Gini = 1 - \sum_{i=1}^{c} p_i^2")
            
            st.markdown("**Entropy:**")
            st.latex(r"H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")
        
        st.success("""
        ### 💡 Quick Summary
        
        | Concept | Simple | Technical |
        |---------|--------|-----------|
        | Supervised Learning | Learn from examples with answers | Learn f(X)→Y from labeled data |
        | Regression | Predict numbers | Continuous output |
        | Classification | Predict categories | Discrete output |
        | Decision Tree | 20 Questions game | Recursive splits to minimize impurity |
        """)
