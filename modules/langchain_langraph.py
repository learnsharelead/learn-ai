import streamlit as st

def show():
    st.title("🔗 LangChain & LangGraph Masterclass")
    
    st.markdown("""
    ### The Most Popular AI Framework
    
    **LangChain** is the go-to library for building LLM-powered applications.
    **LangGraph** extends it for complex, stateful agent workflows.
    """)
    
    tabs = st.tabs([
        "🔗 LangChain Basics",
        "⛓️ Chains",
        "🧠 Memory",
        "🤖 Agents",
        "🔀 LangGraph"
    ])
    
    # TAB 1: LangChain Basics
    with tabs[0]:
        st.header("🔗 What is LangChain?")
        
        st.markdown("""
        ### The "Express.js" of AI
        
        LangChain is a framework that makes it easy to:
        - Connect to any LLM provider
        - Build complex pipelines (chains)
        - Add memory and context
        - Create agents with tools
        """)
        
        st.info("""
        **Install:** `pip install langchain langchain-openai langchain-community`
        """)
        
        st.subheader("Your First LangChain App")
        
        st.code('''
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Create a model
llm = ChatOpenAI(model="gpt-4o")

# 2. Create messages
messages = [
    SystemMessage(content="You are a pirate. Respond in pirate speak."),
    HumanMessage(content="Tell me about Python programming.")
]

# 3. Invoke the model
response = llm.invoke(messages)
print(response.content)
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Key Concepts")
        
        concepts = [
            ("**LLM**", "The language model (OpenAI, Anthropic, etc.)"),
            ("**Prompt Template**", "Reusable prompt with variables"),
            ("**Chain**", "A sequence of operations (prompt → LLM → output)"),
            ("**Agent**", "An LLM that can use tools and make decisions"),
            ("**Memory**", "Stores conversation history"),
            ("**Retriever**", "Fetches relevant documents for RAG"),
        ]
        
        for term, desc in concepts:
            st.markdown(f"- {term}: {desc}")
    
    # TAB 2: Chains
    with tabs[1]:
        st.header("⛓️ Building Chains")
        
        st.markdown("""
        ### Compose LLM Operations
        
        A **Chain** connects multiple steps together using LCEL (LangChain Expression Language).
        """)
        
        st.subheader("Simple Chain: Prompt → LLM → Output")
        
        st.code('''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Create components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains {topic} simply."),
    ("user", "{question}")
])

llm = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

# 2. Chain them together with LCEL (pipe operator)
chain = prompt | llm | output_parser

# 3. Invoke the chain
result = chain.invoke({
    "topic": "machine learning",
    "question": "What is gradient descent?"
})

print(result)
        ''', language="python")
        
        st.success("""
        **LCEL Syntax:** The `|` (pipe) operator chains components together.
        Data flows left to right: `prompt → llm → parser`
        """)
        
        st.markdown("---")
        
        st.subheader("Structured Output Chain")
        
        st.code('''
from langchain_core.pydantic_v1 import BaseModel, Field

# Define output schema
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary")

# Create structured output chain
structured_llm = llm.with_structured_output(MovieReview)

chain = prompt | structured_llm

# Returns a MovieReview object!
review = chain.invoke({"topic": "movies", "question": "Review Inception"})
print(f"Title: {review.title}, Rating: {review.rating}/10")
        ''', language="python")
    
    # TAB 3: Memory
    with tabs[2]:
        st.header("🧠 Conversation Memory")
        
        st.markdown("""
        ### Remember Previous Messages
        
        Without memory, each LLM call is independent. Memory lets you build **chatbots**.
        """)
        
        st.subheader("Buffer Memory")
        
        st.code('''
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create LLM
llm = ChatOpenAI(model="gpt-4o")

# Create message history store (in-memory for demo)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the LLM with message history
chain_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history
)

# Usage
config = {"configurable": {"session_id": "user_123"}}

response1 = chain_with_history.invoke(
    [HumanMessage("My name is Alice")],
    config=config
)

response2 = chain_with_history.invoke(
    [HumanMessage("What is my name?")],  # It remembers!
    config=config
)
        ''', language="python")
        
        st.info("""
        **Memory Types:**
        - `InMemoryChatMessageHistory`: Stores in RAM (demo only)
        - `RedisChatMessageHistory`: For production
        - `SQLChatMessageHistory`: Persist to database
        - Summarization Memory: Compresses old messages
        """)
    
    # TAB 4: Agents
    with tabs[3]:
        st.header("🤖 Building Agents")
        
        st.markdown("""
        ### AI That Can Take Actions
        
        An **Agent** is an LLM that can:
        1. Decide which **tools** to use
        2. Execute tools
        3. Observe results
        4. Decide next steps
        """)
        
        st.warning("Agents are powerful but unpredictable. Use with guardrails!")
        
        st.subheader("Create a Simple Agent")
        
        st.code('''
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use this for calculations."""
    try:
        return str(eval(expression))
    except:
        return "Error: Invalid expression"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create agent
llm = ChatOpenAI(model="gpt-4o")
tools = [calculator, get_current_time]

agent = create_react_agent(llm, tools)

# Run the agent
response = agent.invoke({
    "messages": [("user", "What is 15% of 250? Also, what time is it?")]
})

print(response["messages"][-1].content)
        ''', language="python")
        
        st.success("""
        **The Agent Will:**
        1. Read the question
        2. Decide to call `calculator` for the math
        3. Decide to call `get_current_time` for the time
        4. Combine results into a final answer
        """)
    
    # TAB 5: LangGraph
    with tabs[4]:
        st.header("🔀 LangGraph: Stateful Agents")
        
        st.markdown("""
        ### Beyond Simple Agents
        
        **LangGraph** lets you build complex, multi-step workflows with:
        - Explicit state management
        - Cycles and loops
        - Human-in-the-loop
        - Parallel execution
        """)
        
        st.info("""
        **Install:** `pip install langgraph`
        """)
        
        st.subheader("LangGraph Concepts")
        
        st.markdown("""
        | Concept | Description |
        |---------|-------------|
        | **State** | A dictionary that flows through the graph |
        | **Node** | A function that transforms state |
        | **Edge** | Connects nodes (can be conditional) |
        | **Graph** | The complete workflow |
        """)
        
        st.subheader("Simple LangGraph Workflow")
        
        st.code('''
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define the state
class AgentState(TypedDict):
    messages: list
    next_step: str

# 2. Define nodes (functions)
def analyze_intent(state: AgentState) -> AgentState:
    """Analyze what the user wants."""
    last_message = state["messages"][-1]
    if "weather" in last_message.lower():
        state["next_step"] = "get_weather"
    else:
        state["next_step"] = "general_response"
    return state

def get_weather(state: AgentState) -> AgentState:
    state["messages"].append("The weather is sunny, 22°C")
    return state

def general_response(state: AgentState) -> AgentState:
    state["messages"].append("I can help you with that!")
    return state

# 3. Build the graph
graph = StateGraph(AgentState)

graph.add_node("analyze", analyze_intent)
graph.add_node("weather", get_weather)
graph.add_node("general", general_response)

graph.set_entry_point("analyze")

# 4. Add conditional routing
graph.add_conditional_edges(
    "analyze",
    lambda state: state["next_step"],
    {
        "get_weather": "weather",
        "general_response": "general"
    }
)

graph.add_edge("weather", END)
graph.add_edge("general", END)

# 5. Compile and run
app = graph.compile()

result = app.invoke({
    "messages": ["What's the weather like?"],
    "next_step": ""
})

print(result["messages"])
        ''', language="python")
        
        st.markdown("---")
        
        st.success("""
        **When to Use LangGraph:**
        - Multi-step workflows (research → draft → review → publish)
        - Human-in-the-loop approvals
        - Complex decision trees
        - Long-running async tasks
        
        **When Simple Agents Are Enough:**
        - Single-turn Q&A
        - Simple tool calls
        - Quick prototypes
        """)
