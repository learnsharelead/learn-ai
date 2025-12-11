import streamlit as st

def show():
    st.title("🤖 OpenAI Assistants API")
    
    st.markdown("""
    ### Stateful Agents-as-a-Service
    
    OpenAI's **Assistants API** handles memory, file retrieval, and code execution for you.
    No need to manage your own vector DB or context window.
    """)
    
    tabs = st.tabs([
        "🧠 Concepts",
        "💻 Implementation",
        "📂 File Search & Code"
    ])
    
    # TAB 1: Concepts
    with tabs[0]:
        st.header("🧠 Core Concepts")
        
        concepts = [
            ("🤖 **Assistant**", "The entity with instructions and model (e.g., GPT-4o)."),
            ("🧵 **Thread**", "A conversation session. Stores history automatically."),
            ("🏃 **Run**", "The execution of an Assistant on a Thread."),
            ("💬 **Message**", "Text or files added to the Thread."),
            ("🔧 **Tools**", "Code Interpreter, File Search, Function Calling."),
        ]
        
        for name, desc in concepts:
            st.markdown(f"- {name}: {desc}")
            
        st.info("Key Advantage: **Infinite Context**. You don't manage the context window; OpenAI does.")

    # TAB 2: Implementation
    with tabs[1]:
        st.header("💻 Basic Workflow")
        
        st.code('''
from openai import OpenAI
client = OpenAI()

# 1. Create Assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You help with math. Write code to solve problems.",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)

# 2. Create Thread
thread = client.beta.threads.create()

# 3. Add Message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 2x + 11 = 54"
)

# 4. Run
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 5. Get Messages
if run.status == 'completed':
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages.data[0].content[0].text.value)
        ''', language="python")

    # TAB 3: Advanced
    with tabs[2]:
        st.header("📂 Advanced Tools")
        
        st.subheader("Code Interpreter")
        st.markdown("""
        The Assistant writes and executes real Python code in a sandbox.
        - **Use limits:** Can analyze large CSVs, plot graphs, solve math.
        - **I/O:** Can generate files (PNGs, CSVs) for the user to download.
        """)
        
        st.subheader("File Search (RAG)")
        st.markdown("""
        Upload PDFs/Docs, and the Assistant automatically:
        1. Chunks and embeds them.
        2. Retrieves relevant content.
        3. Cites sources in the answer.
        """)
        
        st.code('''
# Upload file
file = client.files.create(file=open("data.pdf", "rb"), purpose="assistants")

# Update assistant
client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vs.id]}}
)
        ''', language="python")
