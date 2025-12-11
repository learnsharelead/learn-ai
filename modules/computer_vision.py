import streamlit as st
import numpy as np

def show():
    st.title("👁️ Computer Vision & Deep Learning")
    
    st.markdown("""
    When traditional functions fail, **Deep Learning** shines.
    Computer Vision (CV) allows computers to "see" and interpret images.
    """)
    
    tab1, tab2 = st.tabs(["Convolutional Neural Networks (CNN)", "Interactive Filter Demo"])
    
    with tab1:
        st.header("What is a CNN?")
        st.write("A **Convolutional Neural Network** finds features in images (like edges, corners, textures) using 'Filters'.")
        
        st.subheader("The Architecture")
        st.graphviz_chart("""
        digraph CNN {
            rankdir=LR;
            node [shape=box, style=filled, color=lightyellow];
            
            Input [label="Input Image\n(Pixels)"];
            Conv [label="Convolution\n(Find Features)", color=lightblue];
            Pool [label="Pooling\n(Shrink Image)", color=lightpink];
            Flat [label="Flatten\n(Make 1D)"];
            Dense [label="Dense Layers\n(Classify)", color=lightgreen];
            Output [label="Output\n(Cat/Dog)"];
            
            Input -> Conv -> Pool -> Conv -> Pool -> Flat -> Dense -> Output;
        }
        """)
        
        st.info("**Convolution:** Sliding a small window (filter) over the image to detect specific patterns.")

    with tab2:
        st.header("See Like a Machine: The Filter Operation")
        st.write("How do computers detect 'horizontal lines' or 'vertical edges'? They use mathematical filters.")
        
        # Interactive Image Generation
        size = 20
        # Create a simple image (e.g. a cross or square)
        img = np.zeros((size, size))
        img[5:15, 5:15] = 1 # A box in the middle
        
        st.subheader("1. Original Image (Simple Box)")
        st.caption("0 = Black, 1 = White")
        
        # Display as heatmap
        import plotly.express as px
        fig_orig = px.imshow(img, color_continuous_scale='gray', title="Input Pixel Grid")
        st.plotly_chart(fig_orig, use_container_width=True)
        
        st.subheader("2. Apply a Filter (Convolution)")
        
        filter_type = st.radio("Choose a Filter:", ["Vertical Edge Detector", "Horizontal Edge Detector", "Sharpen"])
        
        if filter_type == "Vertical Edge Detector":
            kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        elif filter_type == "Horizontal Edge Detector":
            kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif filter_type == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            
        st.write("Kernel Matrix:")
        st.write(kernel)
        
        # Manual Convolution Implementation for Demo
        # Simple valid padding
        output = np.zeros((size-2, size-2))
        
        for i in range(size-2):
            for j in range(size-2):
                # Element-wise multiplication sum
                region = img[i:i+3, j:j+3]
                output[i, j] = np.sum(region * kernel)
                
        fig_conv = px.imshow(output, color_continuous_scale='RdBu', title=f"Feature Map ({filter_type})")
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.success("""
        **Observe:**
        - **Vertical Edge Filter**: Highlights the Left and Right sides of the box.
        - **Horizontal Edge Filter**: Highlights the Top and Bottom sides.
        
        This is exactly what the first layer of a CNN learns to do!
        """)
