import sys
import joblib
import pandas as pd
import numpy as np
from pipeline import preprocessing
from model.train_models import frequency_train, cost_train
from model.model_evaluate import evaluate_model_cost, evaluate_model_freq
from pipeline.preprocessing import preprocessing


#---------------------------------------------------------------#
#                                Paths                          #
#---------------------------------------------------------------#

sys.path.insert(1, '/model')
sys.path.insert(2, '/pipeline')

#---------------------------------------------------------------#
#                           Preprocessing                       #
#---------------------------------------------------------------#

# Loading data
data = pd.read_csv('training.csv', sep=';')
# Preprocessing
data = preprocessing(data)

PRIME_AVG = sum(np.expm1(data['total_cost'])) / len(data['total_cost'])
N_0 = len(data[data['total_cost'] == 0])
N_1 = len(data['total_cost']) - N_0

#---------------------------------------------------------------#
#                 Frequency prediction training                 #
#---------------------------------------------------------------#

#freq_pipeline = frequency_train(data)
# test_data = pd.read_csv('test_freq.csv')
# x_test = test_data.drop('frequence_claims', axis=1) 
# y_test = test_data['frequence_claims']
# proba, _, report = evaluate_model_freq(freq_pipeline, x_test, y_test)
# print(proba)
# print(report)

#---------------------------------------------------------------#
#                     Cost prediction training                  #
#---------------------------------------------------------------#

#nn_pipeline = cost_train(data)
# test_data = pd.read_csv('test_cost.csv')
# x_test = test_data.drop('target', axis=1) 
# y_test = test_data['target']
# mse = evaluate_model_cost(nn_pipeline, x_test, y_test)
# print(mse)

#---------------------------------------------------------------#
#                            Merging all                        #
#---------------------------------------------------------------#
# Average premium
#ensemble_model = ModelEnsemble(freq_pipeline, nn_pipeline, PRIME_AVG, N_0, N_1)
# Saving the model
#joblib.dump(ensemble_model, 'ensemble_model.joblib')
# model_load = joblib.load('ensemble_model.joblib')

# df = pd.DataFrame(
    # {
        # "Type": ['A'],
        # "Occupation": ['Employed'],
        # "Age": [30],
        # "Group1": [3],
        # "Bonus": [23],
        # "Poldur": [45],
        # "Value": [10000],
        # "Adind": [1],
        # "Density": [100.0],
        # "Exppdays": [365]
    # }
# )

# prime = model_load.transform(df)
# print(prime)