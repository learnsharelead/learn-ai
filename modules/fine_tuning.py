import streamlit as st

def show():
    st.title("🎛️ Fine-Tuning LLMs")
    
    st.markdown("""
    ### Customize Models for Your Domain
    
    **Fine-tuning** adapts a pre-trained model (like Llama 3 or Mistral) to a specific task 
    or dataset. It's how you get a generalist model to become a specialist.
    """)
    
    tabs = st.tabs([
        "🎯 Concepts",
        "📉 PEFT & LoRA",
        "⚒️ How-To Guide",
        "📊 RAG vs Fine-Tuning"
    ])
    
    # TAB 1: Concepts
    with tabs[0]:
        st.header("🎯 What is Fine-Tuning?")
        
        st.info("""
        **Analogy:**
        - **Pre-training:** Teaching a child to read and write (General knowledge).
        - **Fine-tuning:** Teaching a law student specific legal statutes (Specialized knowledge).
        """)
        
        st.subheader("Types of Fine-Tuning")
        
        st.markdown("""
        | Type | Description | Resource Cost |
        |------|-------------|---------------|
        | **Full Fine-Tuning** | Update ALL model weights | 🔴 Very High (Need 8x A100s) |
        | **PEFT (Parameter-Efficient)** | Update only a small subset of weights | 🟢 Low (Run on 1 GPU) |
        | **RLHF/DPO** | Align with human preference (Good/Bad) | 🟡 Medium |
        """)
    
    # TAB 2: PEFT & LoRA
    with tabs[1]:
        st.header("📉 PEFT: Parameter-Efficient Fine-Tuning")
        
        st.markdown("""
        ### The Magic of LoRA (Low-Rank Adaptation)
        
        Instead of retraining the whole huge model, LoRA attaches small "adapter" layers 
        and only trains those.
        
        **Result:** You can fine-tune a 7B model on a single consumer GPU!
        """)
        
        st.subheader("QLoRA: Quantized LoRA")
        st.markdown("""
        Even more efficient!
        1. **Quantize** the base model to 4-bit (shrink memory usage).
        2. Attach **LoRA** adapters.
        3. Train adapters.
        """)
        
        st.code("""
# Comparing Memory Usage (7B Model)

Full Training:     ~112 GB VRAM (Impossible on consumer GPU)
LoRA (16-bit):     ~16 GB VRAM  (High-end consumer GPU)
QLoRA (4-bit):     ~6 GB VRAM   (Gaming Laptop GPU!)
        """, language="text")
    
    # TAB 3: How-To
    with tabs[2]:
        st.header("⚒️ How to Fine-Tune (Code)")
        
        st.markdown("We use the HuggingFace ecosystem: `transformers`, `peft`, `bitsandbytes`, `trl`.")
        
        st.subheader("1. Setup Model (4-bit)")
        st.code('''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-v0.1"

# 4-bit config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
        ''', language="python")
        
        st.subheader("2. Add LoRA Adapters")
        st.code('''
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,            # Rank (higher = more parameters to train)
    lora_alpha=32,   # Scaling factor
    target_modules=["q_proj", "v_proj"], # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print(f"Trainable params: {model.print_trainable_parameters()}")
# Output: "trainable params: 4,194,304 || all params: 7,245,920,256 || trainable%: 0.057"
        ''', language="python")
        
        st.subheader("3. Train (SFT)")
        st.code('''
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_apiargs
)

trainer.train()
        ''', language="python")

    # TAB 4: Compare
    with tabs[3]:
        st.header("📊 RAG vs Fine-Tuning")
        
        st.markdown("""
        ### Which one do I need?
        
        | Feature | RAG (Retrieval) | Fine-Tuning |
        |---------|-----------------|-------------|
        | **Knowledge Source** | External docs (PDFs, DBs) | Internal weights (Learned) |
        | **Accuracy** | High for facts (citations) | Can hallucinate facts |
        | **Updates** | Instant (add new doc) | Slow (re-train model) |
        | **Style/Tone** | Hard to control | Excellent control |
        | **Domain Lingo** | Okay | Excellent |
        | **Cost** | Low (Vector DB) | Medium (Training compute) |
        """)
        
        st.success("""
        **The Golden Rule:**
        - Need to know **NEW FACTS**? Use **RAG**.
        - Need to learn a **NEW BEHAVIOR/STYLE**? Use **Fine-Tuning**.
        - Need both? **Use Both!** (Fine-tune to understand jargon, RAG to get facts).
        """)
