import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

def load_dataset(name):
    if name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif name == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    return None

def generate_linear_data(n_samples=100, noise=10):
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * noise
    return pd.DataFrame({'feature': X.flatten(), 'target': y.flatten()})
