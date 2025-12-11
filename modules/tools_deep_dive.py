import streamlit as st

def show():
    st.title("🛠️ Tools & Frameworks Deep Dive")
    
    st.markdown("""
    ### Beyond LangChain
    
    LangChain is great, but it's not the only game in town. Explore alternative powerful frameworks.
    """)
    
    tabs = st.tabs([
        "🧠 Semantic Kernel",
        "🔠 DSPy",
        "🌾 Haystack",
        "🦙 LlamaIndex"
    ])
    
    # TAB 1: Semantic Kernel
    with tabs[0]:
        st.header("🧠 Microsoft Semantic Kernel")
        
        st.markdown("""
        **"SDK for integrating LLMs with existing code."**
        - **Language:** C# (Native), Python, Java.
        - **Philosophy:** Plugins and Planners. Very production-oriented.
        """)
        
        st.code('''
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_text_completion_service(
    "dv", OpenAIChatCompletion("gpt-4", api_key, org_id)
)

# Define a semantic function
prompt = "Summarize this: {{$input}}"
summarize = kernel.create_semantic_function(prompt)

print(summarize("Semantic Kernel is awesome..."))
        ''', language="python")

    # TAB 2: DSPy
    with tabs[1]:
        st.header("🔠 DSPy (Stanford)")
        
        st.markdown("""
        **"Programming - not prompting."**
        - Automates prompt engineering.
        - You define the **logic** (signatures), DSPy optimizes the **prompts**.
        """)
        
        st.code('''
import dspy

# 1. Configure
dspy.settings.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo'))

# 2. Define Signature (Input -> Output)
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# 3. Compile (Optimize)
program = dspy.Predict(QA)
response = program(question="Where is Paris?")
print(response.answer)
        ''', language="python")
        
        st.success("Use this if you hate tweaking prompt strings manually!")

    # TAB 3: Haystack
    with tabs[2]:
        st.header("🌾 Haystack (Deepset)")
        
        st.markdown("""
        **"Pipelines for Search and LLMs."**
        - Best specifically for **RAG** and **QA** pipelines at scale.
        - Modular "Nodes" connected in a graph.
        """)
    
    # TAB 4: LlamaIndex
    with tabs[3]:
        st.header("🦙 LlamaIndex")
        
        st.markdown("""
        **"The Data Framework for LLMs."**
        - Best for **ingesting usage** data.
        - Excellent connectors (Notion, Slack, SQL, PDF).
        - Advanced indexing structures (Tree index, Keyword index).
        """)
