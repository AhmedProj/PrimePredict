import unittest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
from models import NNetwork, ModelEnsemble, removing_zero_cost, col_type_selector, random_sampling

class TestModelsAndFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.prime_avg = 1000.0
        self.nn_input_size = 10
        self.nn_model = NNetwork(self.nn_input_size)

        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 1, 0, 0]
        })

        self.x = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        self.y = pd.Series([0, 1, 0, 1, 0])

        self.input = np.array([1, 0.1]).reshape(1, -1)

        self.model1 = DummyClassifier(strategy='stratified', random_state=0)
        self.model1 = self.model1.fit(self.x, self.y)
        self.model2 = LinearRegression()
        self.model2 = self.model2.fit(self.x, self.y)
        self.model_ensemble = ModelEnsemble(self.model1, self.model2, self.prime_avg)

    def test_NNetwork(self):
        # Check if the model is initialized correctly
        self.assertEqual(self.nn_model.module[0].in_features, self.nn_input_size)
        self.assertEqual(self.nn_model.module[5].out_features, 1)

        # Check forward pass
        input_tensor = torch.randn(3, self.nn_input_size, dtype=torch.float64)  # Input tensor with batch size 3
        output_tensor = self.nn_model(input_tensor)
        self.assertEqual(output_tensor.shape, (3, 1))

    def test_ModelEnsemble(self):
        # Test transform method
        premium = self.model_ensemble.transform(self.input)
        self.assertIsInstance(premium, float)

    def test_removing_zero_cost(self):
        x_new, y_new = removing_zero_cost(self.df, self.df['target'])
        self.assertNotIn(0, y_new)

    def test_col_type_selector(self):
        cat_variables, num_variables = col_type_selector(self.x)
        self.assertEqual(len(cat_variables), 0)  # No categorical variables in the sample data
        self.assertEqual(len(num_variables), 2)  # Two numerical variables in the sample data

    def test_random_sampling(self):
        x_resampled, y_resampled = random_sampling(self.df, self.df['target'], values=[0, 1], new_sizes=[3, 2])
        self.assertEqual(len(y_resampled), 5)  # Total number of samples should remain the same

if __name__ == '__main__':
    unittest.main()