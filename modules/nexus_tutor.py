import streamlit as st
import time
import random
import os

# Try to import Gemini, fall back to demo mode if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def show():
    """
    Renders the Nexus Tutor AI Chatbot in the sidebar.
    Supports both real Gemini API and demo mode.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Nexus Tutor")
        
        # Check API configuration (Env Var OR Streamlit Secrets)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key and "GEMINI_API_KEY" in st.secrets:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
            
        use_real_ai = GEMINI_AVAILABLE and gemini_api_key
        
        # --- Settings / Model Selector ---
        with st.popover("⚙️ Settings"):
            if use_real_ai:
                st.success("✅ Gemini API Connected")
                model_choice = st.selectbox("AI Model", ["Gemini 1.5 Flash (Fast)", "Gemini 1.5 Pro (Smart)"])
                temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
            else:
                st.warning("⚠️ Demo Mode (No API Key)")
                st.caption("Set GEMINI_API_KEY environment variable to enable real AI.")
                model_choice = "Demo Mode"
                temperature = 0.7
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # --- Chat History Management ---
        if "chat_history" not in st.session_state:
            greeting = "Hello! I'm Nexus Tutor, your AI learning assistant. Ask me anything about AI, ML, or the curriculum!" if use_real_ai else "Hello! I'm in Demo Mode. Ask me basic questions about AI concepts!"
            st.session_state.chat_history = [
                {"role": "assistant", "content": greeting}
            ]
        
        # --- Display Chat ---
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
        
        # --- Chat Input ---
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
                    
                    # Thinking indicator
                    thinking_phrases = [
                        "Analyzing your question...",
                        "Consulting knowledge base...",
                        "Processing query...",
                        "Thinking..."
                    ]
                    response_placeholder.markdown(f"_{random.choice(thinking_phrases)}_")
                    
                    # Generate Response
                    if use_real_ai:
                        response_text = generate_gemini_response(
                            user_input, 
                            model_choice, 
                            temperature,
                            st.session_state.chat_history,
                            gemini_api_key
                        )
                    else:
                        time.sleep(0.8)  # Simulate API delay
                        response_text = generate_mock_response(user_input)
                    
                    # Typing effect
                    full_response = ""
                    for char in response_text:
                        full_response += char
                        if len(full_response) % 3 == 0:
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    
                    response_placeholder.markdown(full_response)
            
            # 3. Save Assistant Message
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

def generate_gemini_response(prompt, model_choice, temperature, chat_history, api_key):
    """
    Generate response using Google Gemini API.
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Select model
        model_name = "gemini-1.5-flash" if "Flash" in model_choice else "gemini-1.5-pro"
        model = genai.GenerativeModel(model_name)
        
        # Build context from chat history (last 5 messages)
        context_messages = chat_history[-5:] if len(chat_history) > 5 else chat_history
        
        # System prompt
        system_context = """You are Nexus Tutor, an expert AI learning assistant for the AI Nexus Academy platform.

Your role:
- Explain AI/ML concepts clearly and concisely
- Provide code examples when relevant (Python, PyTorch, TensorFlow)
- Guide students through complex topics step-by-step
- Reference specific modules in the curriculum when helpful
- Be encouraging and supportive

Keep responses:
- Under 200 words unless explaining complex code
- Formatted with markdown (use **bold**, `code`, bullet points)
- Practical and actionable

Available curriculum modules:
- Fundamentals, Data Preprocessing, Supervised/Unsupervised Learning
- Neural Networks, Computer Vision, NLP, Transformers
- Generative AI, RAG, LangChain, Agentic AI
- MLOps, Fine-Tuning, Testing, Deployment
"""
        
        # Build conversation history for context
        conversation = system_context + "\n\nConversation:\n"
        for msg in context_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n"
        
        conversation += f"User: {prompt}\nAssistant:"
        
        # Generate response
        response = model.generate_content(
            conversation,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=500,
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"⚠️ **API Error**: {str(e)}\n\nFalling back to demo mode. Please check your API key configuration."

def generate_mock_response(prompt):
    """
    Simple keyword-based response generator for Demo Mode.
    """
    p = prompt.lower()
    
    responses = {
        "hello|hi|hey": "Hi there! 👋 Ready to dive into AI? I can help explain concepts like Neural Networks, Transformers, or guide you through the curriculum!",
        
        "neural network|nn": """**Neural Networks** are computational models inspired by the brain! 🧠

**Key Components:**
1. **Input Layer**: Receives data
2. **Hidden Layers**: Extract features (the magic happens here!)
3. **Output Layer**: Makes predictions

**Example (PyTorch):**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
```

Want to build one? Check out the **Neural Networks** module!""",
        
        "transformer|attention": """**Transformers** revolutionized AI! 🚀

**Key Innovation**: Self-Attention mechanism
- Processes entire sequences in parallel (unlike RNNs)
- Captures long-range dependencies
- Powers GPT, BERT, Gemini

**Formula**: Attention(Q,K,V) = softmax(QK^T/√d)V

Introduced in *"Attention Is All You Need"* (2017)

Explore more in the **Advanced NLP** module!""",
        
        "loss function|loss": """**Loss Functions** measure prediction error! 📉

**Common Types:**
- **MSE** (Mean Squared Error): Regression
- **Cross-Entropy**: Classification
- **Huber Loss**: Robust regression

**Goal**: Minimize loss during training!

**Example:**
```python
loss = nn.CrossEntropyLoss()
output = loss(predictions, targets)
```""",
        
        "rag|retrieval": """**RAG (Retrieval-Augmented Generation)** combines search + LLMs! 🔍

**How it works:**
1. User asks question
2. Retrieve relevant docs from vector DB
3. LLM generates answer using retrieved context

**Benefits:**
- Up-to-date information
- Reduces hallucinations
- Cites sources

Check out the **RAG Tutorial** module for hands-on examples!""",
        
        "python|code": """Here's a quick ML snippet! 💻

```python
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

Try the **Code Playground** for interactive coding!""",
        
        "langchain": """**LangChain** simplifies LLM app development! ⛓️

**Core Concepts:**
- **Chains**: Sequence of LLM calls
- **Agents**: LLMs that use tools
- **Memory**: Conversation history
- **Retrievers**: Connect to vector DBs

**Example:**
```python
from langchain.chains import LLMChain

chain = LLMChain(llm=model, prompt=template)
result = chain.run(input="Explain AI")
```

Dive deeper in the **LangChain & LangGraph** module!""",
    }
    
    # Check for keyword matches
    for keywords, response in responses.items():
        if any(kw in p for kw in keywords.split("|")):
            return response
    
    # Default response
    return """That's an interesting question! 🤔

As I'm in **Demo Mode**, my knowledge is limited. For detailed answers, I recommend:

1. **Explore the Curriculum** - Browse relevant modules
2. **Try the Code Playground** - Hands-on learning
3. **Check the Cheat Sheet** - Quick reference

**Tip**: Set up the Gemini API key to unlock my full potential!"""
