"""
Utils.
"""
import mlflow
import random
import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from pipeline.build_pipeline import create_pipeline


def get_model(
    model_name: str, model_version: str
) -> mlflow.pyfunc.PyFuncModel:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.
    Returns:
        model (mlflow.pyfunc.PyFuncModel): The loaded machine learning model.
    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error



class ModelEnsemble(BaseEstimator, TransformerMixin):
    """ Class to estimate the premium value for a given client.
    It first predict the ocurrence of a insurance claims, then it proceed to calculate the premium.
    This class is not trainable, it receives models that have been already trained.
    """
    def __init__(self, model1, model2, prime_avg=50):
        self.model1 = model1
        self.model2 = model2
        self.prime_avg = prime_avg

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
            prime = self.prime_avg + np.expm1(self.model2.predict(X)[0])
        return prime