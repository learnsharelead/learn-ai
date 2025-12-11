import streamlit as st

def show():
    st.title("🤖 Agentic AI: Building Autonomous Agents")
    
    st.markdown("""
    ### AI That Can Think, Plan, and Act
    
    **Agentic AI** systems can autonomously decompose tasks, use tools, and achieve goals.
    They're the future of AI applications.
    """)
    
    tabs = st.tabs([
        "🧠 What Are Agents?",
        "🔧 Tools & Actions",
        "📋 Planning Patterns",
        "🔄 Agent Loops",
        "🛡️ Safety & Guardrails"
    ])
    
    # TAB 1: What Are Agents
    with tabs[0]:
        st.header("🧠 What is an AI Agent?")
        
        st.markdown("""
        ### From Chatbots to Agents
        
        | Feature | Chatbot | Agent |
        |---------|---------|-------|
        | Input/Output | Text in → Text out | Goal in → Actions out |
        | Tools | None | Can use many tools |
        | Autonomy | Responds only | Decides & acts |
        | Memory | Limited | Long-term |
        | Planning | None | Multi-step reasoning |
        """)
        
        st.markdown("---")
        
        st.subheader("The Agent Loop")
        
        st.graphviz_chart("""
        digraph AgentLoop {
            rankdir=TB;
            node [shape=box, style=filled];
            
            Observe [label="1. Observe\\n(Get input/state)", fillcolor=lightblue];
            Think [label="2. Think\\n(Plan next action)", fillcolor=lightgreen];
            Act [label="3. Act\\n(Execute tool)", fillcolor=orange];
            Check [label="4. Check\\n(Goal achieved?)", fillcolor=lightyellow];
            
            Observe -> Think -> Act -> Check;
            Check -> Observe [label="No, continue"];
            Check -> Done [label="Yes, done"];
            
            Done [label="Return Result", fillcolor=lightpink];
        }
        """)
        
        st.success("""
        **Example Agent Task:** "Book me a flight to Tokyo next Friday"
        
        1. **Think:** I need to search flights, then book one
        2. **Act:** Call `search_flights(destination="Tokyo", date="Friday")`
        3. **Observe:** Found 3 flights, cheapest is $800
        4. **Think:** Should I book the cheapest? Yes.
        5. **Act:** Call `book_flight(flight_id="AA123")`
        6. **Done:** "I've booked flight AA123 to Tokyo for $800!"
        """)
    
    # TAB 2: Tools & Actions
    with tabs[1]:
        st.header("🔧 Tools: The Agent's Hands")
        
        st.markdown("""
        ### What Can Agents Do?
        
        Tools give agents the ability to interact with the world.
        """)
        
        tool_categories = [
            ("🔍 **Search & Retrieval**", ["Web search", "Document search (RAG)", "Database queries"]),
            ("📁 **File Operations**", ["Read files", "Write files", "Parse PDFs"]),
            ("🌐 **APIs**", ["REST APIs", "GraphQL", "Webhooks"]),
            ("💻 **Code Execution**", ["Run Python", "Execute SQL", "Shell commands"]),
            ("📧 **Communication**", ["Send emails", "Slack messages", "Create tasks"]),
            ("🖥️ **Browser Automation**", ["Navigate websites", "Fill forms", "Take screenshots"]),
        ]
        
        for category, tools in tool_categories:
            st.markdown(f"{category}")
            for tool in tools:
                st.markdown(f"  - {tool}")
        
        st.markdown("---")
        
        st.subheader("Defining Tools")
        
        st.code('''
from langchain_core.tools import tool

@tool
def search_database(query: str, table: str = "users") -> str:
    """Search the database for matching records.
    
    Args:
        query: The search term
        table: Which table to search (default: users)
    
    Returns:
        Matching records as JSON
    """
    # Your database logic here
    results = db.search(table, query)
    return json.dumps(results)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    
    Args:
        to: Email address
        subject: Email subject
        body: Email body content
    """
    # Your email logic here
    email_service.send(to, subject, body)
    return f"Email sent to {to}"
        ''', language="python")
        
        st.info("""
        **Key Points:**
        - The **docstring** is crucial! The LLM reads it to understand the tool.
        - Use **type hints** for all parameters.
        - Return **strings** that the LLM can interpret.
        """)
    
    # TAB 3: Planning Patterns
    with tabs[2]:
        st.header("📋 Agent Planning Patterns")
        
        st.markdown("""
        ### How Agents Reason
        
        Different strategies for different tasks.
        """)
        
        patterns = [
            {
                "name": "🔄 ReAct (Reason + Act)",
                "desc": "Think step-by-step, act, observe, repeat.",
                "example": "Thought: I need to search for...\nAction: search()\nObservation: Found 3 results\nThought: Now I should...",
                "best_for": "General-purpose agents"
            },
            {
                "name": "📊 Plan-and-Execute",
                "desc": "Create a full plan first, then execute step by step.",
                "example": "Plan:\n1. Search for flights\n2. Compare prices\n3. Book cheapest\n\nExecuting Step 1...",
                "best_for": "Complex, multi-step tasks"
            },
            {
                "name": "🌳 Tree of Thoughts",
                "desc": "Explore multiple reasoning paths, evaluate each.",
                "example": "Path A: Buy vs Path B: Rent\n→ Evaluate both\n→ Choose best",
                "best_for": "Decision-making, creative tasks"
            },
            {
                "name": "🔀 LLM Compiler",
                "desc": "Decompose task into parallel sub-tasks.",
                "example": "Task: Research 3 companies\n→ Spawn 3 parallel research agents\n→ Combine results",
                "best_for": "Parallelizable tasks"
            },
            {
                "name": "🔁 Reflexion",
                "desc": "Agent reflects on failures and improves.",
                "example": "Attempt 1 failed. Reflection: I should have...\nAttempt 2 with improvements...",
                "best_for": "Learning from mistakes"
            }
        ]
        
        for p in patterns:
            with st.expander(p["name"]):
                st.markdown(f"**What:** {p['desc']}")
                st.code(p["example"], language="text")
                st.markdown(f"**Best for:** {p['best_for']}")
    
    # TAB 4: Agent Loops
    with tabs[3]:
        st.header("🔄 Building Agent Loops")
        
        st.markdown("""
        ### Implementation Patterns
        """)
        
        st.subheader("Simple ReAct Agent")
        
        st.code('''
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Use a search API
    return "Search results..."

# Create agent
llm = ChatOpenAI(model="gpt-4o")
tools = [calculator, web_search]

agent = create_react_agent(llm, tools)

# Run
result = agent.invoke({
    "messages": [("user", "What is the population of Japan divided by 1000?")]
})
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Custom Agent Loop")
        
        st.code('''
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
tools = [...]  # Your tools

def agent_loop(user_goal: str, max_iterations: int = 10):
    messages = [{"role": "user", "content": user_goal}]
    
    for i in range(max_iterations):
        # 1. Get LLM decision
        response = llm.invoke(messages, tools=tools)
        
        # 2. Check if done
        if response.content and not response.tool_calls:
            return response.content  # Final answer!
        
        # 3. Execute tools
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            
            # Find and execute tool
            result = execute_tool(tool_name, tool_args)
            
            # Add result to messages
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call["id"]
            })
        
        messages.append(response)
    
    return "Max iterations reached"
        ''', language="python")
    
    # TAB 5: Safety
    with tabs[4]:
        st.header("🛡️ Agent Safety & Guardrails")
        
        st.markdown("""
        ### Agents Can Be Dangerous
        
        Autonomous agents can:
        - Delete files
        - Send emails
        - Spend money
        - Access private data
        - Run in infinite loops
        """)
        
        st.error("**Never deploy agents without guardrails!**")
        
        st.subheader("Safety Measures")
        
        guardrails = [
            ("🔒 **Permission System**", "Require user approval for sensitive actions"),
            ("⏱️ **Timeouts**", "Limit execution time to prevent infinite loops"),
            ("💰 **Budget Caps**", "Limit API calls and spending"),
            ("📝 **Audit Logging**", "Log all actions for review"),
            ("🚫 **Blocklists**", "Prevent certain tools from being called"),
            ("👀 **Human-in-Loop**", "Require approval at key decision points"),
            ("🧪 **Sandboxing**", "Run in isolated environment"),
            ("🔍 **Output Validation**", "Check outputs before acting"),
        ]
        
        for title, desc in guardrails:
            st.markdown(f"- {title}: {desc}")
        
        st.markdown("---")
        
        st.subheader("Human-in-the-Loop Pattern")
        
        st.code('''
def agent_with_approval(user_goal: str):
    plan = agent.plan(user_goal)
    
    print(f"Agent wants to:\\n{plan}")
    
    # Critical actions require approval
    for step in plan.steps:
        if step.is_sensitive:
            approval = input(f"Approve '{step}'? (y/n): ")
            if approval.lower() != 'y':
                return "Cancelled by user"
        
        step.execute()
    
    return plan.result
        ''', language="python")
        
        st.success("""
        **Key Principle:** Start restrictive, loosen gradually.
        
        1. Begin with human approval for everything
        2. Analyze agent behavior over time
        3. Auto-approve safe, common patterns
        4. Keep critical actions manual
        """)
