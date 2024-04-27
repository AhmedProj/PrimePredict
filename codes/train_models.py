import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skorch import NeuralNetRegressor
from build_pipeline import create_pipeline
from models import col_type_selector, random_sampling, removing_zero_cost, NNetwork
import torch.nn as nn
from torch import optim 



def cost_train(df, epochs):
    X = df.drop(["total_cost"], axis=1)
    y = df["total_cost"]

    # Removing zero values from data
    X, y = removing_zero_cost(X, y)
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


    net = NeuralNetRegressor(
        NNetwork(20),
        criterion=nn.MSELoss,
        max_epochs=epochs,
        optimizer=optim.Adam,
        optimizer__lr = 0.0001,
        verbose = 0
    )
    
    # Defining a selector to get the categorical and numerical variables 
    cat_variables, num_variables = col_type_selector(X)
    
    nn_pipeline = create_pipeline(num_variables, cat_variables, net)

    nn_pipeline.fit(X_train, y_train.values.reshape(-1, 1))
    
    return nn_pipeline

    
def frequency_train(df):
    # Resampling
    X = df.drop(["total_cost", "frequence_claims"], axis=1)
    y = df["frequence_claims"]
    
    # Defining a selector to get the categorical and numerical variables 
    cat_variables, num_variables = col_type_selector(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0) 

    # resampling the data
    X_train, y_train = random_sampling(X_train, y_train, values=[0, 1], new_sizes=[10000, 10000])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


    # Model training
    svc = SVC(kernel='poly', degree=3, class_weight='balanced')

    freq_pipeline = create_pipeline(num_variables, cat_variables, svc)
    freq_pipeline.fit(X_train, y_train)

    return freq_pipeline    