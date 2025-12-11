import streamlit as st

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
        "🚀 Advanced RAG"
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
        
        st.info("""
        **Why Not Fine-Tune Instead?**
        - RAG is faster (no training needed)
        - RAG is cheaper (no GPU costs)
        - RAG stays current (update docs anytime)
        - RAG is auditable (shows sources)
        """)
    
    # TAB 2: Embeddings
    with tabs[1]:
        st.header("🔢 Understanding Embeddings")
        
        st.markdown("""
        ### How Machines Understand Text
        
        **Embeddings** convert text into numbers (vectors) that capture meaning.
        Similar texts have similar vectors!
        """)
        
        st.subheader("The Magic of Vector Space")
        
        st.markdown("""
        Each piece of text becomes a point in high-dimensional space:
        
        | Text | Vector (simplified) |
        |------|---------------------|
        | "Happy dog" | [0.9, 0.1, 0.8, ...] |
        | "Joyful puppy" | [0.85, 0.15, 0.78, ...] ← Similar! |
        | "Sad cat" | [0.2, 0.9, 0.3, ...] ← Different |
        """)
        
        st.subheader("Creating Embeddings")
        
        st.code('''
from openai import OpenAI

client = OpenAI()

# Create embedding for a text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is machine learning?"
)

embedding = response.data[0].embedding
print(f"Vector length: {len(embedding)}")  # 1536 dimensions!
print(f"First 5 values: {embedding[:5]}")
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Popular Embedding Models")
        
        models = [
            ("OpenAI text-embedding-3-small", "1536", "Fast, cheap, good quality"),
            ("OpenAI text-embedding-3-large", "3072", "Best quality, higher cost"),
            ("Cohere embed-v3", "1024", "Great for multilingual"),
            ("HuggingFace all-MiniLM-L6-v2", "384", "Free, local, fast"),
            ("Voyage AI voyage-3", "1024", "Best retrieval performance"),
        ]
        
        for model, dims, notes in models:
            st.markdown(f"- **{model}** ({dims} dims): {notes}")
    
    # TAB 3: Vector Stores
    with tabs[2]:
        st.header("🗄️ Vector Databases")
        
        st.markdown("""
        ### Store and Search Embeddings
        
        A **Vector Database** stores embeddings and enables fast similarity search.
        """)
        
        st.subheader("Popular Vector Databases")
        
        dbs = [
            {
                "name": "ChromaDB",
                "type": "Local",
                "best_for": "Prototyping, small projects",
                "code": '''
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_texts(
    texts=["Doc 1 content", "Doc 2 content"],
    embedding=OpenAIEmbeddings()
)

# Search
results = vectorstore.similarity_search("query", k=3)
'''
            },
            {
                "name": "Pinecone",
                "type": "Cloud",
                "best_for": "Production, scalability",
                "code": '''
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("my-index")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings()
)
'''
            },
            {
                "name": "Qdrant",
                "type": "Both",
                "best_for": "Performance, filtering",
                "code": '''
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # or URL

vectorstore = Qdrant.from_texts(
    texts=["..."],
    embedding=OpenAIEmbeddings(),
    client=client,
    collection_name="my_docs"
)
'''
            }
        ]
        
        for db in dbs:
            with st.expander(f"🗄️ {db['name']} ({db['type']})"):
                st.markdown(f"**Best for:** {db['best_for']}")
                st.code(db['code'], language="python")
    
    # TAB 4: Build RAG
    with tabs[3]:
        st.header("🔨 Build a RAG System")
        
        st.markdown("""
        ### Complete RAG Pipeline
        
        Let's build a document Q&A system from scratch!
        """)
        
        st.subheader("Step 1: Load Documents")
        
        st.code('''
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)

# Load from file
loader = TextLoader("docs/policy.txt")
documents = loader.load()

# Load from PDF
# loader = PyPDFLoader("report.pdf")

# Load from web
# loader = WebBaseLoader("https://example.com")
        ''', language="python")
        
        st.subheader("Step 2: Split into Chunks")
        
        st.code('''
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=["\\n\\n", "\\n", ". ", " "]
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
        ''', language="python")
        
        st.subheader("Step 3: Create Vector Store")
        
        st.code('''
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"  # Save locally
)
        ''', language="python")
        
        st.subheader("Step 4: Create RAG Chain")
        
        st.code('''
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt
template = """Answer based on the following context:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
llm = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question!
answer = rag_chain.invoke("What is our refund policy?")
print(answer)
        ''', language="python")
    
    # TAB 5: Advanced RAG
    with tabs[4]:
        st.header("🚀 Advanced RAG Techniques")
        
        st.markdown("""
        ### Beyond Basic RAG
        
        Real-world RAG systems need optimization.
        """)
        
        techniques = [
            {
                "name": "🔍 Hybrid Search",
                "desc": "Combine vector search with keyword search (BM25)",
                "why": "Better for exact matches (names, codes)"
            },
            {
                "name": "📊 Reranking",
                "desc": "Use a second model to rerank retrieved results",
                "why": "Improves relevance of top results"
            },
            {
                "name": "📑 Parent-Child Chunking",
                "desc": "Embed small chunks, retrieve parent documents",
                "why": "More context for the LLM"
            },
            {
                "name": "🔄 Query Expansion",
                "desc": "Generate multiple versions of the query",
                "why": "Find more relevant documents"
            },
            {
                "name": "📝 Self-RAG",
                "desc": "LLM decides when to retrieve and validates answers",
                "why": "Reduces hallucinations"
            },
            {
                "name": "🗂️ Metadata Filtering",
                "desc": "Filter by date, source, category before vector search",
                "why": "Faster, more targeted results"
            }
        ]
        
        for t in techniques:
            with st.expander(t["name"]):
                st.markdown(f"**What:** {t['desc']}")
                st.markdown(f"**Why:** {t['why']}")
        
        st.markdown("---")
        
        st.subheader("RAG Evaluation Metrics")
        
        st.markdown("""
        | Metric | Description |
        |--------|-------------|
        | **Retrieval Precision** | % of retrieved docs that are relevant |
        | **Retrieval Recall** | % of relevant docs that were retrieved |
        | **Answer Relevance** | Does the answer address the question? |
        | **Faithfulness** | Is the answer grounded in retrieved docs? |
        | **Hallucination Rate** | % of answers with made-up info |
        """)
        
        st.info("""
        **Tools for RAG Evaluation:**
        - Ragas
        - TruLens
        - LangSmith
        - Phoenix (Arize)
        """)
