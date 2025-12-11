import streamlit as st

def show():
    st.title("🛠️ AI for Developers: The Complete Stack")
    
    st.markdown("""
    ### Build Production-Grade AI Applications
    
    This track is for **software engineers**, **developers**, and **testers** who want to build
    real-world applications powered by AI. From simple API calls to complex Agentic systems.
    """)
    
    tabs = st.tabs([
        "🌐 The AI Stack",
        "🔌 API Basics",
        "🔧 Function Calling",
        "💬 Chat Patterns",
        "🚀 Best Practices"
    ])
    
    # TAB 1: The AI Stack
    with tabs[0]:
        st.header("🌐 The Modern AI Developer Stack")
        
        st.markdown("""
        ### From Zero to Production
        
        Building AI apps involves layers. Here's the stack:
        """)
        
        st.graphviz_chart("""
        digraph Stack {
            rankdir=TB;
            node [shape=box, style=filled];
            
            App [label="Your Application\\n(Frontend/Backend)", fillcolor=lightblue];
            Framework [label="AI Frameworks\\n(LangChain, LlamaIndex)", fillcolor=lightgreen];
            Provider [label="Model Providers\\n(OpenAI, Gemini, Anthropic)", fillcolor=orange];
            Infra [label="Infrastructure\\n(Vector DBs, GPU Servers)", fillcolor=lightyellow];
            
            App -> Framework [label="uses"];
            Framework -> Provider [label="calls"];
            Provider -> Infra [label="runs on"];
        }
        """)
        
        st.markdown("---")
        
        st.subheader("🧱 Key Components")
        
        components = [
            ("🧠 **LLM Providers**", "OpenAI, Anthropic, Google Gemini, Mistral, Ollama (local)", "The 'brain' that generates text"),
            ("🔗 **Orchestration**", "LangChain, LlamaIndex, Semantic Kernel", "Connect prompts, tools, and memory"),
            ("🗄️ **Vector Databases**", "Pinecone, Weaviate, ChromaDB, Qdrant", "Store embeddings for RAG"),
            ("🔧 **Tools & APIs**", "Function Calling, MCP, Custom Tools", "Let AI interact with the world"),
            ("📊 **Observability**", "LangSmith, Weights & Biases, Phoenix", "Debug and monitor AI systems"),
        ]
        
        for name, examples, desc in components:
            with st.expander(name):
                st.markdown(f"**Examples:** {examples}")
                st.markdown(f"**What it does:** {desc}")
    
    # TAB 2: API Basics
    with tabs[1]:
        st.header("🔌 Calling AI APIs")
        
        st.markdown("""
        ### Your First LLM Call
        
        All AI providers work similarly: send a prompt, get a response.
        """)
        
        st.subheader("OpenAI Example")
        st.code('''
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain recursion in one sentence."}
    ]
)

print(response.choices[0].message.content)
        ''', language="python")
        
        st.subheader("Google Gemini Example")
        st.code('''
import google.generativeai as genai

genai.configure(api_key="...")

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Explain recursion in one sentence.")

print(response.text)
        ''', language="python")
        
        st.subheader("Anthropic Claude Example")
        st.code('''
from anthropic import Anthropic

client = Anthropic(api_key="...")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain recursion in one sentence."}
    ]
)

print(message.content[0].text)
        ''', language="python")
        
        st.info("""
        **Key Concept:** All providers use a **Messages** format:
        - `system`: Sets the AI's behavior
        - `user`: Your input
        - `assistant`: AI's response
        """)
    
    # TAB 3: Function Calling
    with tabs[2]:
        st.header("🔧 Function Calling / Tool Use")
        
        st.markdown("""
        ### Let AI Use Your Code
        
        Instead of just generating text, AI can **call functions** in your codebase!
        """)
        
        st.warning("This is the foundation of **Agentic AI**. Master this first.")
        
        st.subheader("How It Works")
        
        st.markdown("""
        1. You define **tools** (functions with descriptions)
        2. AI decides **which tool to call** based on the user's request
        3. Your code **executes the tool**
        4. AI **summarizes the result**
        """)
        
        st.code('''
# 1. Define your tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # In real app: call weather API
    return f"Weather in {location}: 22°C, Sunny"

# 2. Describe it for the AI
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

# 3. Let AI use it
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# 4. Execute the function call
tool_call = response.choices[0].message.tool_calls[0]
if tool_call.function.name == "get_weather":
    result = get_weather(tool_call.function.arguments["location"])
        ''', language="python")
        
        st.success("""
        **Real-World Uses:**
        - 🔍 Search databases
        - 📧 Send emails
        - 📊 Query analytics
        - 🛒 Place orders
        - 📁 Read/write files
        """)
    
    # TAB 4: Chat Patterns
    with tabs[3]:
        st.header("💬 Chat Application Patterns")
        
        st.markdown("""
        ### Building Conversational AI
        
        Most AI apps are chat-based. Here are the key patterns:
        """)
        
        patterns = [
            {
                "name": "📝 Simple Q&A",
                "desc": "User asks, AI answers. No memory.",
                "use": "FAQ bots, one-shot queries"
            },
            {
                "name": "💭 Chat with Memory",
                "desc": "AI remembers previous messages in the conversation.",
                "use": "Customer support, assistants"
            },
            {
                "name": "📚 RAG (Retrieval)",
                "desc": "AI searches your documents before answering.",
                "use": "Documentation bots, knowledge bases"
            },
            {
                "name": "🤖 Agentic",
                "desc": "AI plans, uses tools, and executes multi-step tasks.",
                "use": "Complex automation, research agents"
            }
        ]
        
        for p in patterns:
            with st.expander(p["name"]):
                st.markdown(f"**What it is:** {p['desc']}")
                st.markdown(f"**Use case:** {p['use']}")
        
        st.markdown("---")
        
        st.subheader("Memory Management")
        
        st.code('''
# Simple memory: keep all messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    ai_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_response})
    print(f"AI: {ai_response}")
        ''', language="python")
        
        st.info("""
        **Challenge:** Message history grows! Solutions:
        - Summarize old messages
        - Use sliding window
        - Store in database
        - Use LangChain memory classes
        """)
    
    # TAB 5: Best Practices
    with tabs[4]:
        st.header("🚀 Production Best Practices")
        
        st.markdown("""
        ### Ship AI Apps That Don't Break
        """)
        
        practices = [
            ("🔒 **Never hardcode API keys**", "Use environment variables or secret managers."),
            ("⏱️ **Set timeouts**", "LLM calls can hang. Always set max wait time."),
            ("🔄 **Retry with backoff**", "APIs fail. Retry 3x with exponential delay."),
            ("📝 **Log everything**", "Log prompts, responses, latency, costs."),
            ("💰 **Track costs**", "LLM APIs are expensive. Monitor usage."),
            ("🧪 **Test prompts**", "Prompt changes break things. Version and test them."),
            ("🚦 **Rate limit users**", "Prevent abuse and cost explosions."),
            ("🛡️ **Validate outputs**", "AI can return garbage. Validate before using."),
        ]
        
        for title, desc in practices:
            st.markdown(f"{title}: {desc}")
        
        st.markdown("---")
        
        st.subheader("Error Handling Template")
        
        st.code('''
import time
from openai import OpenAI, RateLimitError, APIError

def call_llm_with_retry(prompt, max_retries=3):
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                timeout=30  # 30 second timeout
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            wait = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            
        except APIError as e:
            print(f"API Error: {e}")
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Max retries exceeded")
        ''', language="python")
