import streamlit as st
import numpy as np
import plotly.graph_objects as go

def show():
    st.title("🧠 Module 5: Neural Networks")
    
    st.markdown("""
    ### Your Brain, But Made of Math! Simple + Technical Notes
    """)
    
    tabs = st.tabs(["🧠 Brain Inspiration", "⚡ How They Work", "🔗 Deep Learning", "📝 Complete Notes", "🎮 Interactive Demo"])
    
    # TAB 1: Brain Inspiration
    with tabs[0]:
        st.header("🧠 Inspired by Your Brain!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **Your brain has 86 billion neurons**
            
            Each neuron:
            1. 📥 Receives signals
            2. 🤔 Decides: "Important enough?"
            3. 📤 Sends signal forward
            
            AI copies this with math!
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Artificial Neural Network (ANN)**: A computational 
            model inspired by biological neural networks. It consists 
            of interconnected nodes (artificial neurons) organized 
            in layers, which process information using weighted 
            connections and activation functions.
            """)
        
        st.markdown("---")
        
        st.subheader("🧬 Real vs Artificial Neuron")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧬 Biological Neuron
            
            - **Dendrites:** Receive signals
            - **Soma:** Process signals
            - **Axon:** Send output
            - **Synapse:** Connect to others
            
            If signal > threshold → "fires"
            """)
            
        with col2:
            st.markdown("""
            ### 🤖 Artificial Neuron
            
            - **Inputs (x):** Receive numbers
            - **Weights (w):** Importance of each input
            - **Sum:** Weighted sum (Σ wᵢxᵢ)
            - **Activation:** Apply function
            
            If sum > threshold → outputs signal
            """)
        
        st.subheader("📘 The Perceptron (Simplest Neuron)")
        
        st.latex(r"output = f\left( \sum_{i=1}^{n} w_i x_i + b \right)")
        
        st.markdown("""
        Where:
        - **x** = inputs
        - **w** = weights (learned)
        - **b** = bias (threshold adjustment)
        - **f** = activation function
        """)
    
    # TAB 2: How They Work
    with tabs[1]:
        st.header("⚡ How Neural Networks Work")
        
        st.subheader("🍕 The Coffee Decision Analogy")
        
        st.graphviz_chart("""
        digraph Coffee {
            rankdir=LR;
            node [shape=circle, style=filled];
            
            I1 [label="Tired?\\n(+3)", fillcolor=lightyellow];
            I2 [label="Cold?\\n(+1)", fillcolor=lightyellow];
            I3 [label="Like it?\\n(+2)", fillcolor=lightyellow];
            
            N [label="Sum\\n= 6", fillcolor=lightblue, shape=box];
            
            O [label="Coffee!\\n☕", fillcolor=lightgreen];
            
            I1 -> N;
            I2 -> N;
            I3 -> N;
            N -> O [label="6 > 4? Yes!"];
        }
        """)
        
        st.markdown("""
        **Simple:** If (tired + cold + like coffee) > threshold → Get coffee!
        
        **Technical:** weighted sum + bias → activation function → output
        """)
        
        st.markdown("---")
        
        st.subheader("📘 Activation Functions")
        
        st.markdown("""
        **Why needed?** Without activation, neural networks can only learn linear patterns.
        Activation functions add non-linearity to learn complex patterns.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Sigmoid")
            st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
            st.markdown("Output: 0 to 1")
            st.markdown("*Good for probabilities*")
            
        with col2:
            st.markdown("### ReLU")
            st.latex(r"f(x) = max(0, x)")
            st.markdown("Output: 0 to ∞")
            st.markdown("*Most common today*")
            
        with col3:
            st.markdown("### Tanh")
            st.latex(r"tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
            st.markdown("Output: -1 to 1")
            st.markdown("*Centered at 0*")
        
        # Interactive weight demo
        st.markdown("---")
        st.subheader("🎮 Adjust the Weights!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            good_sleep = st.slider("😴 Sleep Quality (1-10):", 1, 10, 7)
            nice_weather = st.slider("☀️ Weather (1-10):", 1, 10, 8)
            good_food = st.slider("🍕 Food (1-10):", 1, 10, 6)
        
        with col2:
            w1 = st.slider("Weight for Sleep:", 0.0, 1.0, 0.4, 0.1)
            w2 = st.slider("Weight for Weather:", 0.0, 1.0, 0.3, 0.1)
            w3 = st.slider("Weight for Food:", 0.0, 1.0, 0.3, 0.1)
        
        weighted_sum = good_sleep * w1 + nice_weather * w2 + good_food * w3
        
        st.markdown(f"""
        ### Calculation:
        - Sleep × {w1} = {good_sleep * w1:.2f}
        - Weather × {w2} = {nice_weather * w2:.2f}
        - Food × {w3} = {good_food * w3:.2f}
        
        **Weighted Sum = {weighted_sum:.2f}**
        """)
        
        if weighted_sum > 7:
            st.success(f"😊 Happy Day! (Score: {weighted_sum:.1f})")
        elif weighted_sum > 5:
            st.info(f"😐 Okay Day (Score: {weighted_sum:.1f})")
        else:
            st.warning(f"😔 Tough Day (Score: {weighted_sum:.1f})")
    
    # TAB 3: Deep Learning
    with tabs[2]:
        st.header("🔗 Layers & Deep Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### 🍕 Simple Version
            
            **More layers = Smarter network**
            
            Like a company with departments:
            - Layer 1: Reception (receives info)
            - Layer 2: Teams (process details)
            - Layer 3: Manager (makes decisions)
            - Layer 4: CEO (final output)
            """)
            
        with col2:
            st.info("""
            ### 📘 Formal Definition
            
            **Deep Learning** uses neural networks with 
            multiple hidden layers to learn hierarchical
            representations. Each layer learns increasingly
            abstract features from the raw input data.
            
            "Deep" = many layers (typically > 3)
            """)
        
        st.subheader("👁️ How AI Sees Images")
        
        st.graphviz_chart("""
        digraph Layers {
            rankdir=LR;
            node [shape=box, style=filled];
            
            L1 [label="Layer 1\\nEdges", fillcolor=lightyellow];
            L2 [label="Layer 2\\nShapes", fillcolor=lightblue];
            L3 [label="Layer 3\\nParts", fillcolor=lightgreen];
            L4 [label="Layer 4\\nObjects", fillcolor=orange];
            L5 [label="Layer 5\\nIdentity", fillcolor=lightpink];
            
            L1 -> L2 -> L3 -> L4 -> L5;
        }
        """)
        
        st.markdown("""
        - **Layer 1:** "I see edges and lines"
        - **Layer 2:** "These make circles and curves"
        - **Layer 3:** "Those are eyes, nose, mouth"
        - **Layer 4:** "That's a face!"
        - **Layer 5:** "That's John!"
        """)
        
        st.markdown("---")
        
        st.subheader("📘 Backpropagation (How Networks Learn)")
        
        with st.expander("📖 The Learning Process"):
            st.markdown("""
            **Forward Pass:**
            1. Input data flows through network
            2. Each layer computes weighted sums + activations
            3. Output layer produces prediction
            
            **Calculate Loss:**
            - Compare prediction to actual answer
            - Loss = how wrong we were
            
            **Backward Pass (Backpropagation):**
            1. Calculate gradients (how to adjust weights)
            2. Use chain rule to propagate error backwards
            3. Update weights: w = w - learning_rate × gradient
            
            **Repeat** thousands of times!
            """)
        
        with st.expander("📖 Mathematical Formula"):
            st.markdown("**Loss Function (e.g., MSE):**")
            st.latex(r"L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
            
            st.markdown("**Weight Update (Gradient Descent):**")
            st.latex(r"w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}")
            
            st.markdown("**Chain Rule (Backprop):**")
            st.latex(r"\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial w_1}")
            
            st.markdown("""
            Where:
            - **η** = learning rate (step size)
            - **∂L/∂w** = gradient of loss w.r.t. weight
            """)
    
    # TAB 4: Complete Notes
    with tabs[3]:
        st.header("📝 Complete Notes (Copy for Reference)")
        
        st.subheader("🔑 Key Definitions")
        
        definitions = {
            "Neural Network": "Computing system inspired by biological brains, using interconnected nodes (neurons) in layers to process information.",
            
            "Neuron (Node)": "Basic computational unit that receives inputs, applies weights, sums them, adds bias, and applies an activation function.",
            
            "Weight": "Learnable parameter that determines the importance of an input. Adjusted during training to minimize error.",
            
            "Bias": "Learnable parameter added to weighted sum. Allows the activation function to shift left or right.",
            
            "Activation Function": "Non-linear function applied to neuron output. Enables network to learn complex patterns (e.g., ReLU, Sigmoid).",
            
            "Layer": "Collection of neurons at the same depth. Types: Input layer, Hidden layer(s), Output layer.",
            
            "Deep Learning": "ML using neural networks with multiple hidden layers to learn hierarchical features automatically.",
            
            "Forward Pass": "Process of passing input through the network to get output/prediction.",
            
            "Backpropagation": "Algorithm to calculate gradients and update weights by propagating error backwards through the network.",
            
            "Gradient Descent": "Optimization algorithm that updates parameters in the direction that minimizes loss function.",
            
            "Learning Rate": "Hyperparameter controlling step size in gradient descent. Too high = overshoot, too low = slow learning.",
            
            "Epoch": "One complete pass through the entire training dataset.",
            
            "Batch Size": "Number of samples processed before weights are updated.",
            
            "Loss Function": "Measures how wrong predictions are. Goal is to minimize this (e.g., MSE, Cross-Entropy).",
        }
        
        for term, definition in definitions.items():
            with st.expander(f"📖 {term}"):
                st.code(f"{term}: {definition}", language="text")
        
        st.subheader("📊 Key Formulas")
        
        with st.expander("📐 All Formulas"):
            st.markdown("**Neuron Computation:**")
            st.latex(r"z = \sum_{i=1}^{n} w_i x_i + b")
            st.latex(r"a = f(z)")
            
            st.markdown("**Sigmoid Activation:**")
            st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
            
            st.markdown("**ReLU Activation:**")
            st.latex(r"ReLU(z) = max(0, z)")
            
            st.markdown("**Cross-Entropy Loss (Classification):**")
            st.latex(r"L = -\sum_{i} y_i \log(\hat{y}_i)")
            
            st.markdown("**MSE Loss (Regression):**")
            st.latex(r"L = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2")
            
            st.markdown("**Gradient Descent Update:**")
            st.latex(r"w = w - \eta \nabla L")
        
        st.subheader("📊 Common Architectures")
        
        st.markdown("""
        | Architecture | Use Case | Key Feature |
        |-------------|----------|-------------|
        | **MLP** | Tabular data | Fully connected layers |
        | **CNN** | Images | Convolutional filters |
        | **RNN** | Sequences | Memory of past inputs |
        | **LSTM** | Long sequences | Long/short-term memory |
        | **Transformer** | NLP, Vision | Attention mechanism |
        """)
    
    # TAB 5: Interactive Demo
    with tabs[4]:
        st.header("🎮 See Neural Networks in Action!")
        
        st.subheader("How AI Recognizes Handwritten Digits")
        
        st.markdown("""
        **The MNIST Dataset:**
        - 28×28 pixel images = 784 input neurons
        - 10 possible outputs (digits 0-9)
        - Network learns to map pixels → digit
        """)
        
        seven_pattern = """
        □ □ □ □ □ □ □
        ■ ■ ■ ■ ■ ■ □
        □ □ □ □ □ ■ □
        □ □ □ □ ■ □ □
        □ □ □ ■ □ □ □
        □ □ □ ■ □ □ □
        □ □ □ ■ □ □ □
        """
        
        st.code(seven_pattern)
        
        st.markdown("""
        **What Each Layer "Sees":**
        
        - **Layer 1:** Detects edges (horizontal line at top, diagonal going down)
        - **Layer 2:** Combines edges into strokes
        - **Layer 3:** Recognizes the "7" shape pattern
        - **Output:** 97% confident it's a "7"!
        """)
        
        st.success("""
        ### 🎉 Key Insight
        
        **Simple:** Neural networks split problems into layers, each finding patterns in the layer before.
        
        **Technical:** Hierarchical feature learning through backpropagation of errors.
        
        This is how your phone reads your handwriting, banks process checks, 
        and post offices sort mail!
        """)
