import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("🗣️ Advanced NLP & Transformers")
    
    st.markdown("""
    Go beyond Bag of Words! Learn modern NLP with Transformers, BERT, and Hugging Face.
    """)
    
    tabs = st.tabs([
        "🔄 Word Embeddings Deep Dive",
        "🤖 Transformers Architecture",
        "🎭 BERT & Fine-tuning",
        "🤗 Hugging Face",
        "😊 Sentiment Analysis Demo"
    ])
    
    # TAB 1: Embeddings
    with tabs[0]:
        st.header("🔄 Word Embeddings Deep Dive")
        
        st.markdown("""
        **Embeddings** map words to dense vectors where similar words are close together.
        """)
        
        st.subheader("Evolution of Word Representations")
        
        st.markdown("""
        | Method | Year | Type | Captures Context? |
        |--------|------|------|-------------------|
        | One-Hot Encoding | - | Sparse | ❌ No |
        | TF-IDF | - | Sparse | ❌ No |
        | Word2Vec | 2013 | Dense | ❌ Static |
        | GloVe | 2014 | Dense | ❌ Static |
        | ELMo | 2018 | Dense | ✅ Yes |
        | BERT | 2018 | Dense | ✅ Yes (bidirectional) |
        | GPT | 2018+ | Dense | ✅ Yes (autoregressive) |
        """)
        
        st.subheader("Word2Vec: Two Architectures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### CBOW (Continuous Bag of Words)")
            st.info("Predict center word from context words")
            st.code("Context: [The, cat, on, mat] → Target: sat")
            
        with col2:
            st.markdown("### Skip-gram")
            st.info("Predict context words from center word")
            st.code("Target: sat → Context: [The, cat, on, mat]")
        
        st.subheader("Famous Word Arithmetic")
        
        st.latex(r"\vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen}")
        
        # Visualization
        st.subheader("2D Embedding Visualization")
        
        # Mock embeddings
        words = {
            'king': [2.5, 3.0], 'queen': [2.0, 2.8], 'man': [1.5, 1.0], 'woman': [1.0, 0.8],
            'apple': [-2.0, -1.5], 'orange': [-1.8, -1.3], 'banana': [-2.2, -1.8],
            'python': [0.5, -2.5], 'java': [0.8, -2.3], 'code': [0.3, -2.0]
        }
        
        df = pd.DataFrame(words).T.reset_index()
        df.columns = ['word', 'x', 'y']
        df['category'] = ['Royalty', 'Royalty', 'Gender', 'Gender', 'Fruit', 'Fruit', 'Fruit', 'Programming', 'Programming', 'Programming']
        
        fig = px.scatter(df, x='x', y='y', text='word', color='category', size_max=20)
        fig.update_traces(textposition='top center', marker=dict(size=15))
        fig.update_layout(title="Word Embeddings in 2D Space")
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Transformers
    with tabs[1]:
        st.header("🤖 Transformer Architecture")
        
        st.markdown("""
        **"Attention Is All You Need" (2017)** - The paper that revolutionized NLP.
        """)
        
        st.subheader("Key Innovation: Self-Attention")
        
        st.markdown("""
        **Problem with RNNs:**
        - Sequential processing (slow)
        - Long-range dependencies hard to learn
        
        **Solution: Attention**
        - Process all tokens in parallel
        - Each token can "attend" to all others
        - O(1) path length between any two positions
        """)
        
        st.subheader("The Attention Formula")
        
        st.latex(r"Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
        
        st.markdown("""
        Where:
        - **Q (Query)**: What am I looking for?
        - **K (Key)**: What do I contain?
        - **V (Value)**: What do I give if matched?
        - **d_k**: Dimension of keys (for scaling)
        """)
        
        st.subheader("Multi-Head Attention")
        
        st.code("""
# Pseudo-code
def multi_head_attention(Q, K, V, num_heads=8):
    heads = []
    for i in range(num_heads):
        # Each head learns different relationships
        head_i = attention(Q @ W_q[i], K @ W_k[i], V @ W_v[i])
        heads.append(head_i)
    
    # Concatenate and project
    return concat(heads) @ W_o
        """, language="python")
        
        st.info("💡 Multi-head = Multiple \"perspectives\" on the same input")
        
        st.subheader("Transformer Block")
        
        st.graphviz_chart("""
        digraph TransformerBlock {
            rankdir=TB;
            node [shape=box, style=filled];
            
            Input [label="Input Embeddings", fillcolor=lightyellow];
            Attn [label="Multi-Head Attention", fillcolor=lightblue];
            Add1 [label="Add & Norm", fillcolor=lightgrey];
            FFN [label="Feed-Forward Network", fillcolor=lightgreen];
            Add2 [label="Add & Norm", fillcolor=lightgrey];
            Output [label="Output", fillcolor=lightyellow];
            
            Input -> Attn;
            Input -> Add1 [style=dashed, label="Residual"];
            Attn -> Add1;
            Add1 -> FFN;
            Add1 -> Add2 [style=dashed, label="Residual"];
            FFN -> Add2;
            Add2 -> Output;
        }
        """)
        
        st.markdown("""
        **Key Components:**
        1. **Residual Connections**: Skip connections for gradient flow
        2. **Layer Normalization**: Stabilize training
        3. **Positional Encoding**: Inject position information (sine/cosine)
        """)
    
    # TAB 3: BERT
    with tabs[2]:
        st.header("🎭 BERT: Bidirectional Encoder Representations")
        
        st.markdown("""
        **BERT** (Google, 2018) = Pre-trained Transformer encoder for NLP tasks.
        """)
        
        st.subheader("BERT vs GPT")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### BERT (Encoder)")
            st.info("""
            - **Bidirectional**: Sees left AND right context
            - **Masked Language Model**: Predict [MASK] tokens
            - Best for: Classification, NER, Q&A
            """)
            
        with col2:
            st.markdown("### GPT (Decoder)")
            st.info("""
            - **Autoregressive**: Only sees left context
            - **Next Token Prediction**: Generate text
            - Best for: Text generation, chat
            """)
        
        st.subheader("BERT Pre-training Tasks")
        
        st.markdown("### 1. Masked Language Model (MLM)")
        st.code("""
Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat  sat on the  mat"

# 15% of tokens are masked during training
        """)
        
        st.markdown("### 2. Next Sentence Prediction (NSP)")
        st.code("""
Input A: "The cat sat on the mat."
Input B: "It was a sunny day."

Label: IsNextSentence? → No (50% are random)
        """)
        
        st.subheader("Fine-tuning BERT")
        
        st.markdown("""
        BERT is a **foundation model**. Fine-tune it for specific tasks:
        
        | Task | Add to BERT | Example |
        |------|-------------|---------|
        | Classification | Linear layer on [CLS] | Sentiment analysis |
        | Token Classification | Linear layer per token | NER |
        | Question Answering | Start/End token predictors | SQuAD |
        """)
        
        st.code("""
from transformers import BertForSequenceClassification, Trainer

# Load pre-trained BERT + classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)

# Fine-tune on your data
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
        """, language="python")
    
    # TAB 4: Hugging Face
    with tabs[3]:
        st.header("🤗 Hugging Face Ecosystem")
        
        st.markdown("""
        **Hugging Face** = The GitHub of Machine Learning. Open-source models, datasets, and tools.
        """)
        
        st.subheader("Key Libraries")
        
        st.markdown("""
        | Library | Purpose |
        |---------|---------|
        | `transformers` | Pre-trained models (BERT, GPT, T5, etc.) |
        | `datasets` | Load datasets easily |
        | `tokenizers` | Fast tokenization |
        | `accelerate` | Multi-GPU/TPU training |
        | `peft` | Parameter-efficient fine-tuning (LoRA) |
        | `diffusers` | Image generation models |
        """)
        
        st.subheader("Quick Start Examples")
        
        st.markdown("### 1. Sentiment Analysis (Zero-shot)")
        
        st.code("""
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")

print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
        """, language="python")
        
        st.markdown("### 2. Text Generation")
        
        st.code("""
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50)

print(result[0]['generated_text'])
        """, language="python")
        
        st.markdown("### 3. Named Entity Recognition")
        
        st.code("""
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
result = ner("Elon Musk founded SpaceX in California.")

# [{'entity_group': 'PER', 'word': 'Elon Musk', ...},
#  {'entity_group': 'ORG', 'word': 'SpaceX', ...},
#  {'entity_group': 'LOC', 'word': 'California', ...}]
        """, language="python")
        
        st.markdown("### 4. Question Answering")
        
        st.code("""
from transformers import pipeline

qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is the capital of France."
)

print(result['answer'])  # "Paris"
        """, language="python")
        
        st.subheader("Hugging Face Hub")
        
        st.markdown("""
        **Models:** 500,000+ pre-trained models
        **Datasets:** 100,000+ datasets
        **Spaces:** Free hosting for demos (like this app!)
        
        [hub.huggingface.co](https://huggingface.co/models)
        """)
    
    # TAB 5: Sentiment Demo
    with tabs[4]:
        st.header("😊 Interactive Sentiment Analysis")
        
        st.markdown("""
        Try sentiment analysis with a simple rule-based approach (no API needed).
        For production, use Hugging Face's `pipeline("sentiment-analysis")`.
        """)
        
        st.subheader("Enter Text to Analyze")
        
        text = st.text_area(
            "Your text:",
            "I absolutely love this new phone! The camera is amazing and the battery life is incredible.",
            height=100
        )
        
        # Rule-based sentiment (for demo without API)
        positive_words = set(['love', 'amazing', 'great', 'excellent', 'wonderful', 'fantastic', 
                             'awesome', 'incredible', 'best', 'perfect', 'happy', 'joy', 'beautiful',
                             'brilliant', 'superb', 'outstanding', 'positive', 'good', 'nice'])
        negative_words = set(['hate', 'terrible', 'awful', 'bad', 'horrible', 'worst', 'poor',
                             'disappointing', 'sad', 'angry', 'frustrated', 'annoying', 'boring',
                             'useless', 'broken', 'waste', 'negative', 'ugly', 'stupid'])
        
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if any(p in w for p in positive_words))
        neg_count = sum(1 for w in words if any(n in w for n in negative_words))
        total = pos_count + neg_count
        
        if total == 0:
            sentiment_score = 0.5
        else:
            sentiment_score = pos_count / total
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sentiment_score > 0.6:
                st.success("😊 **POSITIVE**")
            elif sentiment_score < 0.4:
                st.error("😡 **NEGATIVE**")
            else:
                st.info("😐 **NEUTRAL**")
        
        with col2:
            st.metric("Confidence", f"{abs(sentiment_score - 0.5) * 200:.0f}%")
        
        with col3:
            st.metric("Positive Words", pos_count)
            st.metric("Negative Words", neg_count)
        
        # Word highlighting
        st.subheader("Word Analysis")
        
        highlighted = []
        for word in text.split():
            clean_word = word.lower().strip('.,!?')
            if any(p in clean_word for p in positive_words):
                highlighted.append(f":green[{word}]")
            elif any(n in clean_word for n in negative_words):
                highlighted.append(f":red[{word}]")
            else:
                highlighted.append(word)
        
        st.markdown(" ".join(highlighted))
        st.caption("Green = Positive, Red = Negative")
        
        st.info("""
        💡 **Note:** This is a simple rule-based demo. Real sentiment analysis uses:
        - Pre-trained transformers (BERT, RoBERTa)
        - Context understanding
        - Sarcasm detection
        - Aspect-based sentiment (what specifically is positive/negative)
        """)
