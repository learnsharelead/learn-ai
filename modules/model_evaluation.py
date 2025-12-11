import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def show():
    st.title("📊 Model Evaluation & Metrics")
    
    st.markdown("""
    Building a model is easy. Knowing if it's **actually good** is the hard part.
    This module covers all the metrics you need to evaluate ML models.
    """)
    
    tabs = st.tabs([
        "📈 Classification Metrics",
        "📉 Regression Metrics",
        "🎯 Confusion Matrix",
        "📊 ROC & AUC",
        "🔄 Cross-Validation"
    ])
    
    # TAB 1: Classification Metrics
    with tabs[0]:
        st.header("Classification Metrics")
        
        st.subheader("The Big Four")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Accuracy")
            st.latex(r"\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}")
            st.info("When to use: Balanced classes")
            st.warning("⚠️ Misleading when classes are imbalanced!")
            
            st.markdown("### Precision")
            st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
            st.info("'Of all positive predictions, how many were correct?'")
            st.caption("Use when: False Positives are costly (Spam filter)")
            
        with col2:
            st.markdown("### Recall (Sensitivity)")
            st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
            st.info("'Of all actual positives, how many did we catch?'")
            st.caption("Use when: False Negatives are costly (Cancer detection)")
            
            st.markdown("### F1-Score")
            st.latex(r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
            st.info("Harmonic mean of Precision & Recall")
            st.caption("Use when: You need a balance of both")
        
        st.markdown("---")
        st.subheader("Interactive Example: The Spam Filter")
        
        # Interactive sliders for TP, TN, FP, FN
        st.write("Adjust the confusion matrix values:")
        c1, c2, c3, c4 = st.columns(4)
        TP = c1.number_input("True Positives (Spam → Spam)", 0, 1000, 80)
        TN = c2.number_input("True Negatives (Not Spam → Not Spam)", 0, 1000, 900)
        FP = c3.number_input("False Positives (Not Spam → Spam)", 0, 1000, 10)
        FN = c4.number_input("False Negatives (Spam → Not Spam)", 0, 1000, 10)
        
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Accuracy", f"{accuracy:.2%}")
        met2.metric("Precision", f"{precision:.2%}")
        met3.metric("Recall", f"{recall:.2%}")
        met4.metric("F1-Score", f"{f1:.2%}")
    
    # TAB 2: Regression Metrics
    with tabs[1]:
        st.header("Regression Metrics")
        
        st.subheader("Common Metrics")
        
        st.markdown("### Mean Absolute Error (MAE)")
        st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.info("Average absolute difference. Easy to interpret!")
        
        st.markdown("### Mean Squared Error (MSE)")
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        st.info("Penalizes large errors more heavily (squared).")
        
        st.markdown("### Root Mean Squared Error (RMSE)")
        st.latex(r"RMSE = \sqrt{MSE}")
        st.info("Same units as the target variable. Most popular.")
        
        st.markdown("### R² Score (Coefficient of Determination)")
        st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
        st.info("Proportion of variance explained by the model. 1.0 is perfect.")
        
        # Interactive Demo
        st.markdown("---")
        st.subheader("🎮 Interactive: Visualize Error")
        
        np.random.seed(42)
        X = np.linspace(0, 10, 20)
        y_true = 2 * X + 5 + np.random.normal(0, 2, 20)
        
        slope = st.slider("Adjust Line Slope", 0.0, 4.0, 2.0, 0.1)
        intercept = st.slider("Adjust Line Intercept", 0.0, 10.0, 5.0, 0.5)
        
        y_pred = slope * X + intercept
        
        # Calculate errors
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X, y=y_true, mode='markers', name='Actual', marker=dict(size=10, color='blue')))
        fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='Prediction', line=dict(color='red', width=3)))
        
        # Draw error lines
        for i in range(len(X)):
            fig.add_shape(type="line", x0=X[i], y0=y_true[i], x1=X[i], y1=y_pred[i], line=dict(color="green", width=1, dash="dot"))
        
        fig.update_layout(title="Regression: Actual vs Predicted", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig, use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("MSE", f"{mse:.2f}")
        m3.metric("RMSE", f"{rmse:.2f}")
    
    # TAB 3: Confusion Matrix
    with tabs[2]:
        st.header("🎯 The Confusion Matrix")
        
        st.markdown("""
        A 2x2 table that summarizes True/False Positives/Negatives.
        """)
        
        # Generate sample data
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
        y_pred = y_true.copy()
        # Introduce some errors
        flip_indices = np.random.choice(100, size=15, replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plotly Heatmap
        fig = px.imshow(
            cm, 
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Negative (0)", "Positive (1)"],
            y=["Negative (0)", "Positive (1)"],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(title="Confusion Matrix Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        | | Predicted Negative | Predicted Positive |
        |---|---|---|
        | **Actual Negative** | True Negative (TN) ✅ | False Positive (FP) ❌ |
        | **Actual Positive** | False Negative (FN) ❌ | True Positive (TP) ✅ |
        """)
        
        st.info("""
        **Pro Tip:** FP = "Type I Error" (False Alarm), FN = "Type II Error" (Missed Detection).
        """)
    
    # TAB 4: ROC & AUC
    with tabs[3]:
        st.header("📊 ROC Curve & AUC")
        
        st.markdown("""
        **ROC (Receiver Operating Characteristic)**: Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds.
        
        **AUC (Area Under Curve)**: A single number summarizing the ROC. **AUC = 1.0** is perfect, **AUC = 0.5** is random guessing.
        """)
        
        # Generate data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='gray', dash='dash')))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"**AUC Score: {roc_auc:.3f}**")
        
        st.markdown("""
        | AUC Range | Interpretation |
        |---|---|
        | 0.9 - 1.0 | Excellent |
        | 0.8 - 0.9 | Good |
        | 0.7 - 0.8 | Fair |
        | 0.6 - 0.7 | Poor |
        | < 0.6 | Fail (Random) |
        """)
    
    # TAB 5: Cross-Validation
    with tabs[4]:
        st.header("🔄 Cross-Validation")
        
        st.markdown("""
        **Problem:** A single train/test split can be misleading. What if you got "lucky" with the split?
        
        **Solution:** **K-Fold Cross-Validation** splits data into K parts. Train on K-1, test on 1. Repeat K times.
        """)
        
        st.subheader("Visual: 5-Fold Cross-Validation")
        
        # Visual representation
        folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
        colors = [["Test", "Train", "Train", "Train", "Train"],
                  ["Train", "Test", "Train", "Train", "Train"],
                  ["Train", "Train", "Test", "Train", "Train"],
                  ["Train", "Train", "Train", "Test", "Train"],
                  ["Train", "Train", "Train", "Train", "Test"]]
        
        fig = go.Figure()
        for i, fold in enumerate(folds):
            for j, status in enumerate(colors[i]):
                color = 'lightblue' if status == "Train" else 'salmon'
                fig.add_trace(go.Bar(
                    x=[i], y=[1], name=f'{fold} - Set {j+1}',
                    marker_color=color, showlegend=False,
                    text=status, textposition='inside'
                ))
        
        fig.update_layout(barmode='stack', title="K-Fold Visualization", yaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Each iteration uses a different fold as the test set (red).")
        
        # Live Demo
        st.subheader("🎮 Interactive Demo")
        
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        k = st.slider("Number of Folds (K)", 2, 10, 5)
        
        if st.button("Run Cross-Validation"):
            model = RandomForestClassifier(random_state=42)
            scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
            
            st.write(f"**Scores per Fold:** {[round(s, 3) for s in scores]}")
            st.metric("Mean Accuracy", f"{scores.mean():.2%}")
            st.metric("Standard Deviation", f"{scores.std():.2%}")
            
            fig = go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(k)], y=scores, marker_color='steelblue'))
            fig.add_hline(y=scores.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {scores.mean():.2f}")
            fig.update_layout(title="Cross-Validation Scores", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
