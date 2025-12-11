import streamlit as st
import time
import random

def show():
    """
    Renders the Nexus Tutor AI Chatbot in the sidebar.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Nexus Tutor")
        
        # --- Settings / Model Selector ---
        with st.popover("⚙️ Settings"):
            model_choice = st.selectbox("AI Model", ["Nexus-Lite (Demo)", "Gemini Pro (API)", "Ollama (Local)"])
            st.caption("Currently running in Demo Mode.")
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # --- Chat History Management ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! I'm your AI Tutor. Ask me anything about the curriculum, or paste code to debug!"}
            ]

        # --- Display Chat ---
        # Create a container for the chat messages to keep them organized
        chat_container = st.container()
        
        # Loop through history
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
        
        # --- Chat Input ---
        # Note: In sidebar, input is fixed to bottom
        user_input = st.chat_input("Ask Nexus Tutor...")
        
        if user_input:
            # 1. User Message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)
            
            # 2. AI Response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Mock Thinking Delay
                    thinking_phrases = [
                        "Accessing neural pathways...",
                        "Searching vector database...",
                        "Tokenizing your query...",
                        "Consulting the weights..."
                    ]
                    response_placeholder.markdown(f"_{random.choice(thinking_phrases)}_")
                    time.sleep(0.8)
                    
                    # Generate Response (Mock Logic)
                    # In a real app, this would call OpenAI/Gemini
                    response_text = generate_mock_response(user_input)
                    
                    # simulate typing effect
                    for char in response_text: # Character by character for smoothness
                        full_response += char
                        # Update less frequently to save rendering
                        if len(full_response) % 3 == 0: 
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    
                    response_placeholder.markdown(full_response)
            
            # 3. Save Assistant Message
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def generate_mock_response(prompt):
    """
    Simple keyword-based response generator for Demo Mode.
    """
    p = prompt.lower()
    
    if "hello" in p or "hi" in p:
        return "Hi there! Ready to learn some AI? You can ask me about creating Neural Networks or how Transformers work."
        
    if "neural network" in p:
        return "A Neural Network is a computational model inspired by the human brain. It consists of layers of nodes (neurons) that process data. Key components include:\n\n1. Input Layer\n2. Hidden Layers (where the magic happens)\n3. Output Layer\n\nWould you like to build one in the Lab?"
        
    if "transformer" in p:
        return "Transformers (introduced in 'Attention Is All You Need', 2017) differ from RNNs because they process entire sequences in parallel using 'Self-Attention'. This allows them to capture long-range dependencies effectively. They are the architecture behind GPT and Gemini."
        
    if "loss function" in p:
        return "A loss function measures how well your model predicts the target. Common ones include:\n- MSE (Mean Squared Error) for Regression\n- Cross-Entropy for Classification.\n\nThe goal of training is to minimize this loss!"
        
    if "python" in p or "code" in p:
        return "Here's a simple PyTorch snippet for you:\n```python\nimport torch.nn as nn\n\nmodel = nn.Sequential(\n    nn.Linear(10, 50),\n    nn.ReLU(),\n    nn.Linear(50, 1)\n)\n```"
        
    return "That's an interesting topic! As I'm currently in 'Demo Mode', I have limited knowledge, but I can tell you that mastering the basics of Calculus and Linear Algebra is key to understanding this deeply. Try looking at the 'Core AI' module!"
