import unittest
import pandas as pd
import numpy as np
from utils.data_processing import clean_messy_data, normalize_data

class TestDataProcessing(unittest.TestCase):
    
    def test_clean_messy_data(self):
        # Create sample messy data
        messy_data = pd.DataFrame({
            'Student': ['Alice', 'Bob', 'Dave'],
            'Age': [22, 'twenty', None],
            'Score': [85, 78, None],
            'Grade': ['A', 'C', 'B'],
            'City': ['New York', 'LA', 'nyc']
        })
        
        cleaned = clean_messy_data(messy_data)
        
        # Check Age
        self.assertEqual(cleaned.loc[1, 'Age'], 20)
        self.assertEqual(cleaned.loc[2, 'Age'], 23) # Default fill
        
        # Check City
        self.assertEqual(cleaned.loc[1, 'City'], 'Los Angeles')
        self.assertEqual(cleaned.loc[2, 'City'], 'New York')
        
        # Check Score filled
        self.assertFalse(pd.isna(cleaned.loc[2, 'Score']))
        
    def test_normalize_data(self):
        data = np.array([10, 20, 30])
        normalized = normalize_data(data)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(normalized, expected)

if __name__ == '__main__':
    unittest.main()
