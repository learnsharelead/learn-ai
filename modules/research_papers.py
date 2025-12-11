import streamlit as st
import numpy as np

def show():
    st.title("📄 Research Paper Walkthroughs")
    
    st.markdown("""
    Learn by understanding the landmark papers that shaped modern AI.
    Each walkthrough explains the key ideas, architecture, and impact.
    """)
    
    papers = st.tabs([
        "🔄 Attention Is All You Need",
        "🎭 BERT",
        "🖼️ ResNet",
        "🌀 Generative Adversarial Networks",
        "🎨 Diffusion Models"
    ])
    
    # Paper 1: Attention Is All You Need
    with papers[0]:
        st.header("🔄 Attention Is All You Need (2017)")
        st.markdown("**Authors:** Vaswani et al. (Google) | **Citations:** 100,000+")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 TL;DR")
            st.info("""
            Introduced the **Transformer** architecture - the foundation for GPT, BERT, and all modern LLMs.
            
            **Key Innovation:** Replace RNNs entirely with self-attention mechanism.
            """)
            
        with col2:
            st.metric("Impact Score", "10/10")
            st.metric("Difficulty", "Medium-Hard")
        
        st.subheader("🎯 The Problem")
        
        st.markdown("""
        **Before Transformers (RNNs/LSTMs):**
        - Sequential processing → slow (can't parallelize)
        - Long-range dependencies hard to learn
        - Vanishing/exploding gradients
        
        **The Question:** Can we ditch recurrence entirely?
        """)
        
        st.subheader("💡 The Solution: Self-Attention")
        
        st.markdown("""
        **Self-Attention** lets each token look at ALL other tokens simultaneously.
        
        **The Formula:**
        """)
        
        st.latex(r"Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
        
        st.markdown("""
        **Intuition:**
        - **Query (Q):** "What am I looking for?"
        - **Key (K):** "What do I contain?"
        - **Value (V):** "What do I give back?"
        
        **Example:** In "The animal didn't cross the street because **it** was too tired"
        - "it" attends strongly to "animal" (not "street")
        """)
        
        st.subheader("🏗️ Architecture")
        
        st.graphviz_chart("""
        digraph Transformer {
            rankdir=TB;
            node [shape=box, style=filled];
            
            subgraph cluster_encoder {
                label="Encoder (6 layers)";
                E1 [label="Input Embedding\\n+ Positional Encoding", fillcolor=lightyellow];
                E2 [label="Multi-Head Attention", fillcolor=lightblue];
                E3 [label="Add & Norm", fillcolor=lightgrey];
                E4 [label="Feed Forward", fillcolor=lightgreen];
                E5 [label="Add & Norm", fillcolor=lightgrey];
                E1 -> E2 -> E3 -> E4 -> E5;
            }
            
            subgraph cluster_decoder {
                label="Decoder (6 layers)";
                D1 [label="Output Embedding\\n+ Positional Encoding", fillcolor=lightyellow];
                D2 [label="Masked Multi-Head Attention", fillcolor=lightblue];
                D3 [label="Cross Attention\\n(to Encoder)", fillcolor=orange];
                D4 [label="Feed Forward", fillcolor=lightgreen];
                D5 [label="Linear + Softmax", fillcolor=lightpink];
                D1 -> D2 -> D3 -> D4 -> D5;
            }
            
            E5 -> D3;
        }
        """)
        
        st.subheader("🔑 Key Innovations")
        
        innovations = [
            ("Multi-Head Attention", "8 parallel attention heads, each learning different relationships"),
            ("Positional Encoding", "Sine/cosine functions inject position info (no recurrence!)"),
            ("Residual + LayerNorm", "Every sub-layer has skip connection + normalization"),
            ("Parallelization", "All positions processed simultaneously (100x faster!)"),
        ]
        
        for name, desc in innovations:
            st.markdown(f"**{name}:** {desc}")
        
        st.subheader("📊 Results")
        
        st.success("""
        - **BLEU 28.4** on English-German translation (SOTA at the time)
        - **Training time:** 3.5 days on 8 GPUs (vs weeks for RNNs)
        - **Foundation for:** GPT, BERT, T5, LLaMA, and ALL modern LLMs
        """)
    
    # Paper 2: BERT
    with papers[1]:
        st.header("🎭 BERT: Bidirectional Encoder Representations (2018)")
        st.markdown("**Authors:** Devlin et al. (Google) | **Citations:** 80,000+")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 TL;DR")
            st.info("""
            Pre-trained language model that reads text **bidirectionally**.
            
            **Key Innovation:** Masked Language Modeling (MLM) - predict [MASK] tokens.
            Fine-tune for any NLP task with minimal effort.
            """)
            
        with col2:
            st.metric("Impact Score", "10/10")
            st.metric("Difficulty", "Medium")
        
        st.subheader("🎯 The Problem")
        
        st.markdown("""
        **Before BERT:**
        - Pre-training was unidirectional (left-to-right OR right-to-left)
        - Task-specific architectures needed
        - Limited transfer learning
        
        **BERT's Insight:** Understanding requires BOTH directions!
        
        *"I went to the [BANK] to deposit money"* - need right context!
        *"The river [BANK] was muddy"* - same word, different meaning
        """)
        
        st.subheader("💡 Pre-training Tasks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Masked Language Model (MLM)")
            st.code("""
Input:  "The [MASK] sat on the [MASK]"
Output: "The  cat  sat on the  mat"

# 15% of tokens are masked:
# - 80% replaced with [MASK]
# - 10% random token
# - 10% unchanged
            """)
            
        with col2:
            st.markdown("### Next Sentence Prediction (NSP)")
            st.code("""
Sentence A: "The cat sat on the mat."
Sentence B: "It was a sunny day."

Is B the actual next sentence? → No (50% real, 50% random)
            """)
        
        st.subheader("🏗️ Model Sizes")
        
        st.markdown("""
        | Model | Layers | Hidden | Heads | Parameters |
        |-------|--------|--------|-------|------------|
        | BERT-Base | 12 | 768 | 12 | 110M |
        | BERT-Large | 24 | 1024 | 16 | 340M |
        """)
        
        st.subheader("🔧 Fine-tuning")
        
        st.markdown("""
        BERT can be fine-tuned for almost any NLP task:
        
        | Task | How to Fine-tune |
        |------|------------------|
        | Classification | Add linear layer on [CLS] token |
        | NER | Add linear layer per token |
        | Question Answering | Predict start/end token positions |
        | Sentence Similarity | Compare [CLS] embeddings |
        """)
        
        st.success("""
        **Impact:**
        - SOTA on 11 NLP benchmarks
        - Spawned: RoBERTa, ALBERT, DistilBERT, XLNet, etc.
        - Still used in production everywhere
        """)
    
    # Paper 3: ResNet
    with papers[2]:
        st.header("🖼️ Deep Residual Learning (ResNet) - 2015")
        st.markdown("**Authors:** He et al. (Microsoft) | **Citations:** 170,000+")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 TL;DR")
            st.info("""
            Enabled training of **very deep** neural networks (100+ layers).
            
            **Key Innovation:** Skip/Residual connections that bypass layers.
            """)
            
        with col2:
            st.metric("Impact Score", "10/10")
            st.metric("Difficulty", "Easy-Medium")
        
        st.subheader("🎯 The Problem")
        
        st.markdown("""
        **Degradation Problem:** Deeper networks performed WORSE!
        
        - Not overfitting (training error was also worse)
        - Optimization difficulty: gradients vanish/explode
        - 56-layer network worse than 20-layer
        
        **The Paradox:** A 56-layer network should at least match 20-layer
        (identity mapping for extra layers)
        """)
        
        st.subheader("💡 The Solution: Residual Learning")
        
        st.markdown("Instead of learning H(x), learn F(x) = H(x) - x")
        
        st.latex(r"H(x) = F(x) + x")
        
        st.graphviz_chart("""
        digraph ResBlock {
            rankdir=LR;
            node [shape=box, style=filled];
            
            X [label="x", fillcolor=lightyellow];
            Conv1 [label="Conv\\nBN\\nReLU", fillcolor=lightblue];
            Conv2 [label="Conv\\nBN", fillcolor=lightblue];
            Add [label="+", shape=circle, fillcolor=lightgreen];
            ReLU [label="ReLU", fillcolor=lightblue];
            Out [label="H(x)", fillcolor=lightyellow];
            
            X -> Conv1 -> Conv2 -> Add;
            X -> Add [label="identity (skip)", style=dashed];
            Add -> ReLU -> Out;
        }
        """)
        
        st.markdown("""
        **Why it works:**
        - Easy to learn identity: just set F(x) = 0
        - Gradients flow directly through skip connections
        - Each layer refines, rather than transforms
        """)
        
        st.subheader("📊 Results")
        
        st.markdown("""
        | Model | Layers | Top-5 Error |
        |-------|--------|-------------|
        | VGG-19 | 19 | 7.5% |
        | ResNet-34 | 34 | 5.7% |
        | ResNet-152 | 152 | **3.6%** |
        
        **Won ImageNet 2015** with 3.57% error (vs 7.5% human-level)
        """)
        
        st.success("""
        **Legacy:**
        - Foundation for all modern CNNs
        - Skip connections used everywhere (DenseNet, U-Net, Transformers!)
        - Enabled training of 1000+ layer networks
        """)
    
    # Paper 4: GANs
    with papers[3]:
        st.header("🌀 Generative Adversarial Networks (2014)")
        st.markdown("**Authors:** Goodfellow et al. | **Citations:** 60,000+")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 TL;DR")
            st.info("""
            Two networks compete: Generator creates fakes, Discriminator detects them.
            
            **Key Innovation:** Adversarial training for generative models.
            """)
            
        with col2:
            st.metric("Impact Score", "10/10")
            st.metric("Difficulty", "Medium")
        
        st.subheader("💡 The Concept")
        
        st.markdown("""
        **The Analogy:**
        - 🎨 **Generator:** Art forger creating fake paintings
        - 🔍 **Discriminator:** Art detective identifying fakes
        
        They compete until the forger is perfect!
        """)
        
        st.graphviz_chart("""
        digraph GAN {
            rankdir=LR;
            node [shape=box, style=filled];
            
            Noise [label="Random\\nNoise (z)", fillcolor=lightyellow];
            G [label="Generator", fillcolor=lightblue];
            Fake [label="Fake\\nImage", fillcolor=lightpink];
            Real [label="Real\\nImage", fillcolor=lightgreen];
            D [label="Discriminator", fillcolor=orange];
            Out [label="Real or\\nFake?", fillcolor=lightyellow];
            
            Noise -> G -> Fake -> D;
            Real -> D;
            D -> Out;
            
            Out -> G [label="Train G", style=dashed];
            Out -> D [label="Train D", style=dashed];
        }
        """)
        
        st.subheader("📐 The Math")
        
        st.markdown("**Minimax Game:**")
        
        st.latex(r"\min_G \max_D V(D, G) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]")
        
        st.markdown("""
        - **D wants:** Maximize (correctly classify real vs fake)
        - **G wants:** Minimize (fool D with better fakes)
        
        **Nash Equilibrium:** G produces perfect fakes, D guesses 50/50
        """)
        
        st.subheader("🎨 GAN Variants")
        
        st.markdown("""
        | Variant | Innovation | Application |
        |---------|------------|-------------|
        | DCGAN | Convolutional architecture | Image generation |
        | cGAN | Conditional generation | Class-specific output |
        | CycleGAN | Unpaired image translation | Horse↔Zebra |
        | StyleGAN | Disentangled style control | Photorealistic faces |
        | WGAN | Wasserstein distance | Stable training |
        """)
        
        st.success("""
        **Impact:**
        - Revolutionized image generation
        - DeepFakes (controversial)
        - Data augmentation
        - Art and design tools
        - Largely superseded by Diffusion Models (2022+)
        """)
    
    # Paper 5: Diffusion Models
    with papers[4]:
        st.header("🎨 Denoising Diffusion Probabilistic Models (2020)")
        st.markdown("**Authors:** Ho et al. (Google) | **Key Paper for DALL-E, Stable Diffusion, Midjourney**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 TL;DR")
            st.info("""
            Gradually destroy images with noise, then learn to reverse the process.
            
            **Key Innovation:** Iterative denoising produces high-quality images.
            Powers DALL-E 2, Stable Diffusion, Midjourney!
            """)
            
        with col2:
            st.metric("Impact Score", "10/10")
            st.metric("Difficulty", "Hard")
        
        st.subheader("💡 The Concept")
        
        st.markdown("""
        **Forward Process (Fixed):**
        Gradually add Gaussian noise until image becomes pure noise.
        
        **Reverse Process (Learned):**
        Train a neural network to predict and remove the noise step by step.
        """)
        
        st.subheader("🔄 The Process")
        
        # Timeline visualization
        st.markdown("""
        ```
        Forward (destroy):
        [Clean Image] → [Slightly Noisy] → [More Noisy] → ... → [Pure Noise]
             x_0      →       x_1        →     x_2      → ... →    x_T
        
        Reverse (generate):
        [Pure Noise] → [Less Noisy] → [Cleaner] → ... → [Generated Image!]
             x_T      →    x_{T-1}   →   x_{T-2} → ... →      x_0
        ```
        """)
        
        st.subheader("📐 Key Equations")
        
        st.markdown("**Forward Process:**")
        st.latex(r"q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)")
        
        st.markdown("**Training Objective (simplified):**")
        st.latex(r"L = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]")
        
        st.markdown("Predict the noise $\\epsilon$ added at step $t$.")
        
        st.subheader("🚀 Why Diffusion > GANs?")
        
        st.markdown("""
        | Aspect | GANs | Diffusion |
        |--------|------|-----------|
        | Training | Unstable (mode collapse) | Stable |
        | Diversity | Can miss modes | Full coverage |
        | Quality | Good | **State-of-the-art** |
        | Controllability | Hard | Easy (classifier-free guidance) |
        | Speed | Fast generation | Slow (many steps) |
        """)
        
        st.subheader("🛠️ Key Innovations for Scale")
        
        st.markdown("""
        1. **Latent Diffusion (Stable Diffusion):** Work in compressed latent space → 10x faster
        2. **Classifier-Free Guidance:** Trade diversity for quality with a single parameter
        3. **ControlNet:** Add precise spatial control (pose, edges, depth)
        4. **LoRA:** Efficient fine-tuning for custom styles
        """)
        
        st.success("""
        **Impact:**
        - DALL-E 2, Midjourney, Stable Diffusion
        - Text-to-video (Sora)
        - Image editing
        - 3D generation
        - The AI art revolution!
        """)
        
        st.info("""
        **Further Reading:**
        - "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
        - "Classifier-Free Diffusion Guidance"
        - "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
        """)
