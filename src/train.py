import sys
import joblib
import pandas as pd
import numpy as np
from pipeline import preprocessing
from model.models import ModelEnsemble
from model.train_models import frequency_train, cost_train
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
data = preprocessing(data)

PRIME_AVG = sum(np.expm1(data['total_cost'])) / len(data['total_cost'])

#---------------------------------------------------------------#
#                 Frequency prediction training                 #
#---------------------------------------------------------------#

freq_pipeline = frequency_train(data)

#---------------------------------------------------------------#
#                     Cost prediction training                  #
#---------------------------------------------------------------#

nn_pipeline  = cost_train(data)

#---------------------------------------------------------------#
#                            Merging all                        #
#---------------------------------------------------------------#
# Average premium
ensemble_model = ModelEnsemble(freq_pipeline, nn_pipeline, PRIME_AVG)
# Saving the model
joblib.dump(ensemble_model, 'ensemble_model.joblib')
