import pandas as pd
import random
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn as nn
from build_pipeline import create_pipeline

    
class NNetwork(nn.Module):
    def __init__(self, input_size):
        super(NNetwork, self).__init__()
        
        self.module = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        )
        self.double()
        
    def forward(self,x):
        x = self.module(x)
        return x

class ModelEnsemble(BaseEstimator, TransformerMixin):
    """ With this class we can estimate the premium value for a given client.
    We first calculate the frequency of accidents, then we proceed to calculate the premium.
    This class is not trainable, it receives models that have been already trained.
    """
    def __init__(self, model1, model2, prime_avg):
        self.model1 = model1
        self.model2 = model2
        self.prime_avg = prime_avg

    def fit(self, X, y=None):
        return self

    def adding_extra_cols(self, X):
        """Function to add the dummies features to X
        """
        cols = list(X.index)
        values = list(X)
        missing_cols = ['Type_A', 'Occupation_Employed', 'Occupation_Retired', 
                        'Occupation_Unemployed', 'Type_D', 'Occupation_Self-employed', 'Type_F', 
                        'Type_B', 'Occupation_Housewife', 'Type_E', 'Type_C', 'Type', 'Occupation'
                        ]
             
        new_cols = list(set(missing_cols) - set(cols))
        cols.extend(new_cols)
        values.extend(len(new_cols) * [0])

        test_df = pd.DataFrame(columns=cols)
        test_df.loc[0] = values

        # Recovering right format
        n_a, n_b = test_df['Type'], test_df['Occupation']
        test_df.drop(['Type', 'Occupation'], axis=1, inplace=True)
        test_df['Type_' + n_a] = 1
        test_df['Occupation_' + n_b] = 1

        return test_df

    def transform(self, X):
        """Method to calculate the premium value for a given set of parameters X.
        PARAMETERS
        ----------
        X: DataFrame
           Dataframe containing the features
        OUTPUT
        ------
        premium: float
               premium value predicted
        """
        X = self.adding_extra_cols(X)
        model1_output = self.model1.predict(X)

        if model1_output == 0:
            prime = self.prime_avg
        else:
            X["frequence_claims"] = model1_output
            prime = self.prime_avg + np.expm1(self.model2.predict(X)[0, 0])
        return prime

def removing_zero_cost(x, y):
    """
    Function to remove the values zero from the target value 
    """
    x_new = pd.DataFrame()
    x['target'] = y
    x_new = x[x['target'] != 0]
    return x_new.drop(['target'], axis=1), x_new['target']

def col_type_selector(X):

    categorical_selector = selector(dtype_include = object)
    numerical_selector = selector(dtype_exclude=object)    
    cat_variables = categorical_selector(X)
    num_variables = numerical_selector(X)
    return cat_variables, num_variables

def random_sampling(x, y, values=[1], new_sizes=[0]):
    """
    This function performs oversampling or undersampling,
    depending on the class size and the requested new_size

    PARAMETERS
    ----------
    x: DataFrame
       Dataframe containing the features
    y: Series
       1D array with axis labels that contains the different classes
    values: List of integers    
            It contains the class values required to resample 
    new_sizes: List of integers
               size required for the corresponding class in values
    OUTPUT
    ------
    x_result: DataFrame
              Resampled dataframe containing the features
    y_result: Series
              Resampled series object containing the different classes
    """

    x['target'] = y
    for val, size in zip(values, new_sizes):
        df_sampled = x[x['target'] == val]
        n_lines = df_sampled.shape[0]
        # Over_sampling
        if n_lines <= size:
            rdn_rows = random.choices(range(0, n_lines), k=size - df_sampled.shape[0])
            x = pd.concat([x, df_sampled.iloc[rdn_rows]], ignore_index=True)
        # Under_sampling    
        else:    
            rdn_rows = random.sample(list(df_sampled.index), k=df_sampled.shape[0] - size)
            x = x.drop(rdn_rows)

    x_result = x.drop('target', axis=1)
    y_result = x['target']   
    
    return x_result, y_result    