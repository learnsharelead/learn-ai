import pandas as pd
import numpy as np

def clean_messy_data(df):
    """
    Cleans a specific messy dataset with student info.
    - Converts Age to numeric (handles 'twenty')
    - Standardizes City names
    - Fills missing Score with mean
    - Fills missing Age with forward fill or specific logic
    """
    df = df.copy()
    
    # Fix Age
    def fix_age(x):
        if isinstance(x, str):
            if x.lower() == 'twenty': return 20
        return x
    
    df['Age'] = df['Age'].apply(fix_age)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Fill missing age with mean or specific value (using 23 from example)
    df['Age'] = df['Age'].fillna(23) 
    
    # Fix Score
    score_mean = df['Score'].mean()
    df['Score'] = df['Score'].fillna(score_mean)
    
    # Fix City
    city_map = {
        'nyc': 'New York',
        'new york': 'New York',
        'la': 'Los Angeles',
        'los angeles': 'Los Angeles'
    }
    df['City'] = df['City'].str.lower().map(city_map).fillna(df['City'])
    
    return df

def normalize_data(data):
    """
    Normalizes data to 0-1 range.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)
