import unittest
import pandas as pd
import numpy as np
from utils.data_generators import generate_linear_data, load_dataset

class TestUtils(unittest.TestCase):
    
    def test_generate_linear_data(self):
        df = generate_linear_data(n_samples=50)
        self.assertEqual(len(df), 50)
        self.assertTrue('feature' in df.columns)
        self.assertTrue('target' in df.columns)
        
    def test_load_dataset_iris(self):
        df = load_dataset("iris")
        self.assertIsNotNone(df)
        self.assertTrue('target' in df.columns)
        self.assertEqual(len(df.columns), 5) # 4 features + 1 target

    def test_load_dataset_invalid(self):
        df = load_dataset("invalid_name")
        self.assertIsNone(df)

if __name__ == '__main__':
    unittest.main()
