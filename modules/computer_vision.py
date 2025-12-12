import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
import io

def show():
    st.title("👁️ Computer Vision & Deep Learning")
    
    st.markdown("""
    **Computer Vision (CV)** enables machines to interpret and understand visual information from the world.
    From simple edge detection to complex object recognition, CV powers self-driving cars, medical diagnosis, and facial recognition.
    """)
    
    tabs = st.tabs([
        "🧠 CNN Architecture",
        "🔍 Filter Demo",
        "📸 Image Classifier",
        "🔄 Transfer Learning",
        "📦 Object Detection",
        "🎨 Data Augmentation"
    ])
    
    # TAB 1: CNN Architecture
    with tabs[0]:
        st.header("🧠 Convolutional Neural Networks (CNN)")
        
        st.markdown("""
        **Why CNNs?** Traditional neural networks treat images as flat arrays, losing spatial relationships.
        CNNs preserve the 2D structure and learn hierarchical features.
        """)
        
        st.subheader("The Architecture")
        st.graphviz_chart("""
        digraph CNN {
            rankdir=LR;
            node [shape=box, style=filled, color=lightyellow];
            
            Input [label="Input Image\n(224x224x3)", color=lightcyan];
            Conv1 [label="Conv Layer 1\n(Find Edges)", color=lightblue];
            Pool1 [label="MaxPool\n(Downsample)", color=lightpink];
            Conv2 [label="Conv Layer 2\n(Find Shapes)", color=lightblue];
            Pool2 [label="MaxPool", color=lightpink];
            Conv3 [label="Conv Layer 3\n(Find Objects)", color=lightblue];
            Flat [label="Flatten\n(Vector)"];
            Dense [label="Dense Layers\n(Classify)", color=lightgreen];
            Output [label="Output\n(1000 classes)"];
            
            Input -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3 -> Flat -> Dense -> Output;
        }
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Key Operations:**
            
            1. **Convolution**: Applies filters to detect features
            2. **Activation (ReLU)**: Introduces non-linearity
            3. **Pooling**: Reduces spatial dimensions
            4. **Fully Connected**: Final classification
            """)
        
        with col2:
            st.markdown("""
            **Feature Hierarchy:**
            
            - **Layer 1**: Edges, colors, gradients
            - **Layer 2**: Textures, simple shapes
            - **Layer 3**: Object parts (eyes, wheels)
            - **Layer 4+**: Complete objects
            """)
        
        st.info("💡 **Key Insight**: CNNs automatically learn features from data, unlike hand-crafted features in traditional CV.")

    # TAB 2: Filter Demo
    with tabs[1]:
        st.header("🔍 Interactive Filter Visualization")
        st.write("Experience how CNNs 'see' by applying convolution filters to images.")
        
        # Interactive Image Generation
        size = 20
        img = np.zeros((size, size))
        
        pattern = st.selectbox("Choose Pattern:", ["Box", "Cross", "Diagonal", "Circle"])
        
        if pattern == "Box":
            img[5:15, 5:15] = 1
        elif pattern == "Cross":
            img[10, :] = 1
            img[:, 10] = 1
        elif pattern == "Diagonal":
            np.fill_diagonal(img, 1)
        elif pattern == "Circle":
            y, x = np.ogrid[:size, :size]
            mask = (x - 10)**2 + (y - 10)**2 <= 25
            img[mask] = 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            fig_orig = px.imshow(img, color_continuous_scale='gray', title="Input")
            st.plotly_chart(fig_orig, use_container_width=True)
        
        with col2:
            st.subheader("After Convolution")
            
            filter_type = st.radio("Filter:", ["Vertical Edge", "Horizontal Edge", "Sharpen", "Blur"])
            
            if filter_type == "Vertical Edge":
                kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            elif filter_type == "Horizontal Edge":
                kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            elif filter_type == "Sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            else:  # Blur
                kernel = np.ones((3, 3)) / 9
            
            # Convolution
            output = np.zeros((size-2, size-2))
            for i in range(size-2):
                for j in range(size-2):
                    output[i, j] = np.sum(img[i:i+3, j:j+3] * kernel)
            
            fig_conv = px.imshow(output, color_continuous_scale='RdBu', title=f"{filter_type} Output")
            st.plotly_chart(fig_conv, use_container_width=True)
        
        with st.expander("🔬 View Kernel Matrix"):
            st.write(kernel)

    # TAB 3: Image Classifier
    with tabs[2]:
        st.header("📸 Live Image Classification")
        st.markdown("""
        Upload an image and watch a **pre-trained MobileNetV2** model classify it in real-time.
        This model was trained on **ImageNet** (1.4M images, 1000 categories).
        """)
        
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("🤖 AI Prediction")
                
                # Simulate classification (in production, use TensorFlow/PyTorch)
                st.info("⚠️ **Demo Mode**: Using simulated predictions. In production, this would use TensorFlow.js or a backend API.")
                
                # Mock predictions
                predictions = [
                    ("Golden Retriever", 0.87),
                    ("Labrador", 0.09),
                    ("Beagle", 0.03),
                    ("Poodle", 0.01)
                ]
                
                for label, conf in predictions:
                    st.progress(conf, text=f"{label}: {conf:.1%}")
                
                st.success(f"**Top Prediction**: {predictions[0][0]} ({predictions[0][1]:.1%} confidence)")
        
        else:
            st.info("👆 Upload an image to see the classifier in action!")
        
        with st.expander("🧠 How Does This Work?"):
            st.markdown("""
            **MobileNetV2 Architecture:**
            1. **Input**: 224x224x3 RGB image
            2. **Backbone**: 53 convolutional layers with depthwise separable convolutions
            3. **Output**: 1000-class probability distribution
            
            **Why MobileNet?**
            - Lightweight (14MB vs 500MB for ResNet)
            - Fast inference (mobile-friendly)
            - 71% top-1 accuracy on ImageNet
            
            **Real Implementation:**
            ```python
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.preprocessing import image
            
            model = MobileNetV2(weights='imagenet')
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            ```
            """)

    # TAB 4: Transfer Learning
    with tabs[3]:
        st.header("🔄 Transfer Learning")
        
        st.markdown("""
        **The Problem**: Training a CNN from scratch requires millions of images and weeks of GPU time.
        
        **The Solution**: Use a pre-trained model and fine-tune it for your specific task.
        """)
        
        st.subheader("Transfer Learning Workflow")
        
        st.graphviz_chart("""
        digraph TL {
            rankdir=TB;
            node [shape=box, style=filled];
            
            Pre [label="Pre-trained Model\n(ImageNet)", color=lightblue];
            Freeze [label="Freeze Early Layers\n(Keep learned features)", color=lightyellow];
            Replace [label="Replace Final Layer\n(Your classes)", color=lightgreen];
            Train [label="Train on Your Data\n(Fine-tune)", color=lightpink];
            Deploy [label="Deploy Model", color=lightcyan];
            
            Pre -> Freeze -> Replace -> Train -> Deploy;
        }
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Why It Works:**
            - Early layers learn universal features (edges, textures)
            - Only task-specific layers need retraining
            - Requires 10-100x less data
            """)
        
        with col2:
            st.markdown("""
            **Use Cases:**
            - Medical imaging (X-rays, MRIs)
            - Custom object detection
            - Satellite imagery analysis
            - Quality control in manufacturing
            """)
        
        st.code("""
# Transfer Learning Example (PyTorch)
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task (e.g., 10 classes)
model.fc = nn.Linear(2048, 10)

# Train only the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        """, language="python")

    # TAB 5: Object Detection
    with tabs[4]:
        st.header("📦 Object Detection")
        
        st.markdown("""
        **Classification** answers: "What is in this image?"  
        **Object Detection** answers: "What and where are the objects?"
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Popular Architectures")
            st.markdown("""
            **YOLO (You Only Look Once)**
            - Real-time detection (30+ FPS)
            - Single-pass architecture
            - Great for video streams
            
            **Faster R-CNN**
            - Higher accuracy
            - Two-stage detection
            - Better for static images
            
            **EfficientDet**
            - Best accuracy/speed tradeoff
            - Scalable architecture
            """)
        
        with col2:
            st.subheader("Key Concepts")
            st.markdown("""
            **Bounding Boxes**: (x, y, width, height)
            
            **Anchor Boxes**: Pre-defined box shapes
            
            **Non-Max Suppression**: Remove duplicate detections
            
            **IoU (Intersection over Union)**: Measures box overlap
            """)
        
        st.info("💡 **Modern Trend**: Vision Transformers (ViT) are replacing CNNs for state-of-the-art results.")

    # TAB 6: Data Augmentation
    with tabs[5]:
        st.header("🎨 Data Augmentation")
        
        st.markdown("""
        **Problem**: Deep learning needs lots of data. What if you only have 1,000 images?
        
        **Solution**: Create synthetic variations through augmentation.
        """)
        
        # Create a simple demo image
        demo_img = np.random.rand(100, 100, 3)
        demo_img[30:70, 30:70] = [1, 0, 0]  # Red square
        
        st.subheader("Interactive Augmentation Demo")
        
        aug_type = st.selectbox("Augmentation:", [
            "Original",
            "Horizontal Flip",
            "Rotation",
            "Brightness Adjustment",
            "Zoom",
            "Gaussian Noise"
        ])
        
        if aug_type == "Original":
            result = demo_img
        elif aug_type == "Horizontal Flip":
            result = np.fliplr(demo_img)
        elif aug_type == "Rotation":
            from scipy.ndimage import rotate
            result = rotate(demo_img, 45, reshape=False)
        elif aug_type == "Brightness Adjustment":
            result = np.clip(demo_img * 1.5, 0, 1)
        elif aug_type == "Zoom":
            result = demo_img[10:90, 10:90]
        else:  # Gaussian Noise
            noise = np.random.normal(0, 0.1, demo_img.shape)
            result = np.clip(demo_img + noise, 0, 1)
        
        fig = px.imshow(result, title=aug_type)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Common Augmentations:**
        - Geometric: Flip, Rotate, Crop, Zoom
        - Color: Brightness, Contrast, Saturation
        - Noise: Gaussian, Salt & Pepper
        - Advanced: Cutout, Mixup, CutMix
        """)
        
        st.code("""
# Data Augmentation with PyTorch
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
        """, language="python")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Next Steps
    - **Practice**: Try Kaggle's [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) competition
    - **Learn More**: [CS231n Stanford Course](http://cs231n.stanford.edu/)
    - **Build**: Create a custom image classifier for your own dataset
    """)
