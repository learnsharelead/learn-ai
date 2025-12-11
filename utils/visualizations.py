import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_scatter(data, x, y, color=None, title="Scatter Plot"):
    fig = px.scatter(data, x=x, y=y, color=color, title=title)
    return fig

def plot_loss_curve(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='loss'))
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='val_loss'))
    fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
    return fig
