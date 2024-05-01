import random
import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from pipeline.build_pipeline import create_pipeline


class ModelEnsemble(BaseEstimator, TransformerMixin):
    """ Class to estimate the premium value for a given client.
    It first predict the ocurrence of a insurance claims, then it proceed to calculate the premium.
    This class is not trainable, it receives models that have been already trained.
    """
    def __init__(self, model1, model2, prime_avg=86.76, n0=42294, n1=5751):
        self.model1 = model1
        self.model2 = model2
        self.prime_avg = prime_avg
        self.n0 = n0
        self.n1 = n1        

    def fit(self, X, y=None):
        return self

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
        model1_output = self.model1.predict(X)
        if model1_output == 0:
            prime = self.prime_avg
        else:
            X["frequence_claims"] = model1_output
            prime = self.prime_avg + self.n1 * np.expm1(self.model2.predict(X)[0]) / self.n0
        return round(prime, 2)

def removing_zero_cost(x, y):
    """
    Function to remove the values zero from the target value
    """
    x_new = pd.DataFrame()
    x['target'] = y
    x_new = x[x['target'] != 0]
    return x_new.drop(['target'], axis=1), x_new['target']

def col_type_selector(X):
    """Function to get the name of categorical and numerical features.
    PARAMETERS
    ----------
    X: DataFrame
       Dataframe containing the features
    OUTPUT
    ------
    cat_variables: List of strings
                   Name of columns with categorical features
    num_variables: List of strings
                   Name of columns with numerical features           
    """
    categorical_selector = selector(dtype_include=object)
    numerical_selector = selector(dtype_exclude=object)
    cat_variables = categorical_selector(X)
    num_variables = numerical_selector(X)
    return cat_variables, num_variables

def random_sampling(x, y, values, new_sizes):
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
