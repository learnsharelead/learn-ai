import streamlit as st

def show():
    st.title("👥 Multi-Agent Systems (CrewAI & AutoGen)")
    
    st.markdown("""
    ### Teamwork Makes the Dream Work
    
    Why use one agent when you can have a team? **Multi-Agent Systems** orchestrated specialized agents
    to solve complex problems like software development, research, or content creation.
    """)
    
    tabs = st.tabs([
        "🧠 Concepts",
        "🛶 CrewAI",
        "🤖 AutoGen",
        "⚖️ Comparison"
    ])
    
    # TAB 1: Concepts
    with tabs[0]:
        st.header("🧠 Core Concepts")
        
        st.info("**The Manager-Worker Pattern:** Just like a company, one 'Manager' agent delegates tasks to specialized 'Worker' agents.")
        
        st.markdown("""
        **Key Roles:**
        - **Researcher:** Searches the web for info.
        - **Writer:** Drafts content based on research.
        - **Editor:** Reviews and critiques content.
        - **Manager:** Coordinates the workflow.
        
        **Why Multi-Agent?**
        - **Specialization:** Each agent has a focused system prompt.
        - **Parallelism:** Agents can work simultaneously.
        - **Quality:** Evaluation/Critique loops improve output.
        """)

    # TAB 2: CrewAI
    with tabs[1]:
        st.header("🛶 CrewAI")
        
        st.markdown("**Role-playing agents that work together.**")
        
        st.code('''
from crewai import Agent, Task, Crew, Process

# 1. Define Agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI',
    backstory="You are a veteran analyst at a top tech firm.",
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role='Tech Writer',
    goal='Write compelling articles about AI tech',
    backstory="You simplify complex topics for a general audience.",
    verbose=True
)

# 2. Define Tasks
task1 = Task(
    description='Search for the latest AI news from 2024.',
    agent=researcher
)

task2 = Task(
    description='Write a blog post based on the research.',
    agent=writer
)

# 3. Form Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential  # Run one after another
)

# 4. Kickoff
result = crew.kickoff()
print(result)
        ''', language="python")

    # TAB 3: AutoGen
    with tabs[2]:
        st.header("🤖 Microsoft AutoGen")
        
        st.markdown("**Conversable agents that chat to solve tasks.**")
        
        st.code('''
import autogen

# 1. Configuration
config_list = [{"model": "gpt-4", "api_key": "..."}]

# 2. Define Agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="TERMINATE"
)

# 3. Start Chat
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA stock price YTD."
)
        ''', language="python")
        
        st.success("AutoGen is amazing for **Code Generation** because the proxy actively executes the code locally.")

    # TAB 4: Compare
    with tabs[3]:
        st.header("⚖️ CrewAI vs AutoGen")
        
        st.markdown("""
        | Feature | CrewAI | AutoGen |
        |---------|--------|---------|
        | **Focus** | Process & Roles (Production) | Conversational Problem Solving (Research) |
        | **Structure** | Rigid, process-driven | Flexible, chat-driven |
        | **Ease of Use** | ⭐⭐⭐⭐⭐ (Easy) | ⭐⭐⭐ (Complex) |
        | **Code Exec** | Via Tools | Native/Local |
        | **Best For** | Content pipelines, Research | Coding, Data Analysis |
        """)
