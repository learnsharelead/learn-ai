import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import random

from utils.audio_manager import listen_section

def show():
    st.title("🧠 Module 5: Neural Networks & Deep Learning")
    
    intro_content = """
    ### From Biological Neurons to PyTorch Code
    Deep learning powers ChatGPT, Midjourney, and Self-Driving Cars. Let's understand how.
    """
    listen_section("Neural Nets Intro", intro_content.replace("#", ""))
    st.markdown(intro_content)
    
    tabs = st.tabs([
        "🧠 Theory", 
        "⚡ How It Works", 
        "🔥 PyTorch Lab", 
        "📝 Notes", 
        "🎮 Weights Demo"
    ])
    
    # TAB 1: Theory
    with tabs[0]:
        st.header("🧠 Inspired by the Brain")
        
        theory_content = """
        Thinking about how Biological Neurons work helps us build Artificial ones.
        
        **Biological Neuron:**
        - **Dendrites**: Receive signals
        - **Soma**: Process signals
        - **Axon**: Send output
        
        **Artificial Neuron:**
        - **Inputs (x)**: The data features
        - **Weights (w)**: The importance of each feature
        - **Bias (b)**: The activation threshold
        """
        listen_section("Theory: Bio vs Artificial", theory_content.replace("*", ""))
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            ### 🧬 Biological Neuron
            - **Dendrites**: Receive signals
            - **Soma**: Process signals
            - **Axon**: Send output
            - **Synapse**: Connect to others
            """)
        with col2:
            st.info("""
            ### 🤖 Artificial Neuron
            - **Inputs (x)**: Features
            - **Weights (w)**: Importance
            - **Bias (b)**: Threshold
            - **Activation**: Non-linearity
            """)
            
        st.latex(r"y = f(\sum w_i x_i + b)")
        st.caption("The math that drives modern AI.")

    # TAB 2: How It Works
    with tabs[1]:
        st.header("⚡ The Mechanics of Learning")
        
        st.subheader("1. Forward Pass (Prediction)")
        st.markdown("Data flows through layers. `Input -> Hidden -> Output`")
        
        st.subheader("2. Loss Calculation (Error)")
        st.markdown("Compare prediction to actual target. `MSE = (Pred - Actual)²`")
        
        st.subheader("3. Backward Pass (Backprop)")
        st.markdown("Calculate gradients: *How much did each weight contribute to the error?*")
        
        st.subheader("4. Optimization (Update)")
        st.markdown("Adjust weights slightly opposite to the error.")
        st.latex(r"w_{new} = w_{old} - \text{learning\_rate} \times \text{gradient}")

    # TAB 3: PyTorch Lab (NEW FEATURE)
    with tabs[2]:
        st.header("🔥 PyTorch Lab")
        st.markdown("### Build a Neural Network from Scratch")
        
        st.markdown("""
        We will build a simple network to learn a relationship: `y = 2x + 10`.
        The model starts strictly random and *learns* the math.
        """)
        
        # Code Display
        code = '''
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data (Teach it: y = 2x + 10)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[12.0], [14.0], [16.0], [18.0]])

# 2. Define Model (1 input -> 1 output)
model = nn.Linear(1, 1)

# 3. Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training Loop
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        '''
        st.code(code, language="python")
        
        # Interactive Simulation
        st.subheader("🚀 Run Training Simulation")
        
        if st.button("Start Training Model"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_place = st.empty()
            
            # Simulation Data
            losses = []
            epochs = []
            
            # True relation: y = 2x + 10
            # Start random: y = 0.5x + 0
            w = 0.5
            b = 0.0
            target_w = 2.0
            target_b = 10.0
            lr = 0.05
            
            for i in range(100):
                # Fake training physics
                w += (target_w - w) * lr
                b += (target_b - b) * lr
                
                # Calculate "Loss" (distance from truth)
                current_loss = abs(target_w - w) + abs(target_b - b) + (random.random() * 0.1)
                
                losses.append(current_loss)
                epochs.append(i)
                
                # Update UI
                if i % 5 == 0:
                    status_text.text(f"Epoch {i}/100 | Loss: {current_loss:.4f}")
                    progress_bar.progress(i + 1)
                    
                    # Plot Loss Curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=epochs, y=losses, mode='lines', name='Loss'))
                    fig.update_layout(title="Training Loss (Lower is Better)", height=300)
                    chart_place.plotly_chart(fig, use_container_width=True)
                    
                time.sleep(0.02) # Fast visual
            
            progress_bar.progress(100)
            st.success("Training Complete!")
            
            st.markdown("### 🎯 Final Prediction")
            st.markdown("The model has learned the pattern `2x + 10`.")
            
            col1, col2 = st.columns(2)
            with col1:
                test_val = st.number_input("Test Input (x)", value=5.0)
            with col2:
                # Prediction based on learned weights
                pred = test_val * w + b
                st.metric("Model Prediction (y)", f"{pred:.2f}", delta=f"True Answer: {test_val * 2 + 10}")

    # TAB 4: Notes
    with tabs[3]:
        st.header("📝 Complete Notes")
        st.markdown("""
        **Key Concepts:**
        1. **Tensor**: A multi-dimensional array (matrix) that runs on GPU.
        2. **Autograd**: PyTorch's engine that calculates gradients automatically.
        3. **Epoch**: One complete pass through the dataset.
        4. **Batch**: A small chunk of data processed at once.
        5. **Learning Rate**: Step size for updates. Too big = unstable, too small = slow.
        """)
        
        st.subheader("Common Layers")
        st.code("""
nn.Linear(in, out)    # Dense layer
nn.Conv2d(...)        # Convolution (Images)
nn.LSTM(...)          # Recurrent (Text/Time)
nn.Dropout(p=0.5)     # Regularization (Prevents overfitting)
        """, language="python")

    # TAB 5: Weights Demo
    with tabs[4]:
        st.header("🎮 Neural Playground")
        st.markdown("Adjust weights to see how neurons fire!")
        
        input_val = st.slider("Input Signal", -10.0, 10.0, 2.0)
        weight = st.slider("Weight", -5.0, 5.0, 1.0)
        bias = st.slider("Bias", -5.0, 5.0, 0.0)
        
        act = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
        
        z = (input_val * weight) + bias
        
        if act == "ReLU":
            a = max(0, z)
            func_eq = "max(0, z)"
        elif act == "Sigmoid":
            a = 1 / (1 + np.exp(-z))
            func_eq = "1 / (1 + e⁻ᶻ)"
        else:
            a = np.tanh(z)
            func_eq = "tanh(z)"
            
        st.latex(f"Output = {act}(({input_val} \\times {weight}) + {bias})")
        st.latex(f"Output = {act}({z:.2f}) = {a:.4f}")
        
        st.metric("Neuron Output", f"{a:.4f}")
