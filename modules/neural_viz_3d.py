import streamlit as st
import plotly.graph_objects as go
import numpy as np
import networkx as nx

def show():
    st.header("🧊 3D Neural Network Visualizer")
    st.markdown("Interact with the architecture of deep learning headers in 3D space. Rotate, zoom, and explore layers!")

    col_config, col_viz = st.columns([1, 3])

    with col_config:
        st.subheader("🏗️ Architecture")
        input_size = st.slider("Input Neurons", 2, 10, 4)
        hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
        hidden_neurons = st.slider("Neurons per Hidden Layer", 2, 10, 5)
        output_size = st.slider("Output Neurons", 1, 5, 2)
        
        st.markdown("---")
        st.caption("Visualization Settings")
        show_weights = st.checkbox("Show Weights (Edges)", value=True)
        theme_color = st.color_picker("Neuron Color", "#00aaff")

    with col_viz:
        # Generate 3D Coordinates
        node_x = []
        node_y = []
        node_z = []
        node_color = []
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        # Helper to add layer
        layers = [input_size] + [hidden_neurons]*hidden_layers + [output_size]
        
        layer_dist = 2.0
        max_neurons = max(layers)
        
        # Calculate positions
        positions = {} # (layer_idx, neuron_idx) -> (x, y, z)
        
        for l_idx, n_count in enumerate(layers):
            x = l_idx * layer_dist
            
            # center vertically and depth-wise
            y_start = -(n_count - 1) / 2.0
            
            for n_idx in range(n_count):
                y = y_start + n_idx
                # Zig-zag z for 3D effect if needed, purely flat for now or random
                z = np.sin(n_idx) * 0.5 if n_count > 1 else 0
                
                positions[(l_idx, n_idx)] = (x, y, z)
                
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                # Color logic
                if l_idx == 0:
                    node_color.append('#e91e63') # Input: Pink
                elif l_idx == len(layers)-1:
                    node_color.append('#00e676') # Output: Green
                else:
                    node_color.append(theme_color) # Hidden: User choice
        
        # Calculate Edges
        if show_weights:
            for l_idx in range(len(layers) - 1):
                curr_layer_count = layers[l_idx]
                next_layer_count = layers[l_idx + 1]
                
                for i in range(curr_layer_count):
                    for j in range(next_layer_count):
                        p1 = positions[(l_idx, i)]
                        p2 = positions[(l_idx+1, j)]
                        
                        edge_x.extend([p1[0], p2[0], None])
                        edge_y.extend([p1[1], p2[1], None])
                        edge_z.extend([p1[2], p2[2], None])

        # Create Plotly 3D Scatter
        fig = go.Figure()
        
        # Edges (Lines)
        if show_weights:
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.3)', width=2),
                hoverinfo='none'
            ))

        # Nodes (Spheres)
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=12,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            hoverinfo='text',
            text=[f"Layer {l}<br>Neuron {n}" for l, n in positions.keys()]
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Tip: Click and drag to rotate the network in 3D space. Scroll to zoom.")
