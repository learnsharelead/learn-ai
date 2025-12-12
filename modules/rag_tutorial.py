import streamlit as st
import os
import tempfile
import time
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback to prevent crash during installation
        class RecursiveCharacterTextSplitter:
            def __init__(self, **kwargs): pass
            def split_documents(self, docs): return []
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def show():
    st.title("📚 RAG: Retrieval-Augmented Generation")
    
    st.markdown("""
    ### Give Your AI a Knowledge Base
    
    **RAG** is the technique that lets AI answer questions about *your* documents,
    not just what it learned during training.
    """)
    
    tabs = st.tabs([
        "🎯 What is RAG?",
        "🔢 Embeddings",
        "🗄️ Vector Stores",
        "🔨 Build RAG",
        "🚀 Advanced RAG",
        "⚡ Live RAG Lab"
    ])
    
    # TAB 1: What is RAG
    with tabs[0]:
        st.header("🎯 What is RAG?")
        
        st.markdown("""
        ### The Problem
        
        LLMs have a **knowledge cutoff**. They don't know about:
        - Your company's internal docs
        - Recent news
        - Personal files
        - Private databases
        
        ### The Solution: RAG
        
        Before answering, **retrieve** relevant information and **augment** the prompt!
        """)
        
        st.graphviz_chart("""
        digraph RAG {
            rankdir=LR;
            node [shape=box, style=filled];
            
            User [label="User Query", fillcolor=lightblue];
            Retriever [label="1. Retriever\\n(Search Docs)", fillcolor=lightgreen];
            Context [label="2. Context\\n(Relevant Chunks)", fillcolor=lightyellow];
            LLM [label="3. LLM\\n(Generate)", fillcolor=orange];
            Answer [label="4. Answer", fillcolor=lightblue];
            
            User -> Retriever;
            Retriever -> Context;
            Context -> LLM;
            LLM -> Answer;
        }
        """)
        
        st.success("""
        **RAG in Simple Terms:**
        1. User asks: "What's our refund policy?"
        2. System searches your docs for "refund policy"
        3. Finds: "Refunds within 30 days..."
        4. Sends to LLM: "Based on this: [refund policy text], answer: What's our refund policy?"
        5. LLM answers accurately!
        """)
    
    # TAB 2: Embeddings
    with tabs[1]:
        st.header("🔢 Understanding Embeddings")
        st.markdown("""
        **Embeddings** convert text into numbers (vectors) that capture meaning.
        Similar texts have similar vectors!
        """)
        
        st.info("Example: 'Dog' and 'Puppy' will have vectors that are numerically close, while 'Dog' and 'Sandwich' will be far apart.")
        
        st.code('''
# Example Vector Representation (Simplified)
"King"   -> [0.9, 0.1, 0.5]
"Queen"  -> [0.9, 0.2, 0.5]
"Apple"  -> [0.1, 0.9, 0.2]
        ''', language="python")

    # TAB 3: Vector Stores
    with tabs[2]:
        st.header("🗄️ Vector Databases")
        st.markdown("A **Vector Database** stores these embeddings for fast retrieval.")
        st.markdown("""
        - **ChromaDB**: Open source, local, easy.
        - **Pinecone**: Cloud-managed, scalable.
        - **Qdrant**: High performance, Rust-based.
        """)

    # TAB 4: Build RAG (Code)
    with tabs[3]:
        st.header("🔨 Build a RAG System")
        st.code('''
# 1. Load
loader = TextLoader("data.txt")
docs = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(docs)

# 3. Embed & Store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 4. Retrieve & Generate
retriever = vectorstore.as_retriever()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm
)
rag_chain.invoke("My question")
        ''', language="python")

    # TAB 5: Advanced
    with tabs[4]:
        st.header("🚀 Advanced RAG")
        st.markdown("- **Hybrid Search**: Keywords + Vectors")
        st.markdown("- **Re-ranking**: Double-check relevance")
        st.markdown("- **Parent Document**: Retrieve full context")

    # TAB 6: Live Lab
    with tabs[5]:
        st.header("⚡ Live RAG Lab")
        st.markdown("Upload a PDF and chat with it!")
        
        uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load & Split
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)
                    
                    st.success(f"✅ Loaded {len(docs)} pages and split into {len(chunks)} chunks.")
                    
                    # Show preview
                    with st.expander("👀 View Document Chunks"):
                        for i, chunk in enumerate(chunks[:3]):
                            st.caption(f"Chunk {i+1}:")
                            st.text(chunk.page_content[:300] + "...")
                    
                    # Interactive Chat
                    st.markdown("### 💬 Chat with your Document")
                    
                    user_q = st.text_input("Ask a question about this document:")
                    
                    if user_q:
                        # SIMULATED RAG (for stability without needing keys in demo)
                        # In production, we would use Chroma + OpenAI/Gemini Embeddings
                        
                        st.markdown("**1. Retrieving relevant chunks...**")
                        
                        # Simple retrieval simulation (Keyword matching)
                        relevant_chunks = []
                        keywords = user_q.lower().split()
                        
                        for chunk in chunks:
                            score = 0
                            content_lower = chunk.page_content.lower()
                            for k in keywords:
                                if k in content_lower:
                                    score += 1
                            if score > 0:
                                relevant_chunks.append((score, chunk.page_content))
                        
                        # Sort by score
                        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
                        top_contexts = [c[1] for c in relevant_chunks[:3]]
                        
                        if not top_contexts:
                            # Fallback if no keywords found (just take first 2)
                            top_contexts = [c.page_content for c in chunks[:2]]
                            st.warning("⚠️ Low relevance match. Using partial context.")
                        
                        context_text = "\n\n".join(top_contexts)
                        
                        st.info(f"🔍 Found {len(top_contexts)} relevant context clips.")
                        with st.expander("View Retrieved Context"):
                            st.markdown(context_text)
                            
                        # Use Nexus Tutor Logic (Gemini) if available
                        st.markdown("**2. Generating Answer...**")
                        
                        prompt = f"""
                        You are a helpful RAG assistant. Answer the question based ONLY on the context provided below.
                        
                        Context:
                        {context_text}
                        
                        Question: {user_q}
                        
                        Answer:
                        """
                        
                        # Check for Gemini
                        api_key = os.getenv("GEMINI_API_KEY")
                        if not api_key and "GEMINI_API_KEY" in st.secrets:
                            api_key = st.secrets["GEMINI_API_KEY"]
                            
                        if api_key:
                            try:
                                import google.generativeai as genai
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-1.5-flash')
                                response = model.generate_content(prompt)
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"AI Generation failed: {str(e)}")
                        else:
                            st.warning("⚠️ No API Key found. Displaying Mock Response.")
                            st.write(f"**Mock Answer:** Based on the document, I found information relevant to '{user_q}'. The document mentions: {context_text[:100]}...")
                            
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.info("Please ensure 'pypdf' is installed in requirements.txt")
                finally:
                    os.unlink(tmp_path)
