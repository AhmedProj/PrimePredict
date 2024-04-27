import os
import pickle
import pandas as pd
import numpy as np
from preprocessing import preprocessing
from train_models import frequency_train, cost_train
from models import ModelEnsemble


#---------------------------------------------------------------#
#                           Preprocessing                       #
#---------------------------------------------------------------#

# Loading data
cdir = os.getcwd()
path = os.path.dirname(cdir)
data = pd.read_csv(path + '/training.csv', sep=';')
data = preprocessing(data)

PRIME_AVG = sum(np.expm1(data['total_cost'])) / len(data['total_cost'])
EPOCHS = 100

#---------------------------------------------------------------#
#                 Frequency prediction training                 #
#---------------------------------------------------------------#

freq_pipeline = frequency_train(data)

#---------------------------------------------------------------#
#                     Cost prediction training                  #
#---------------------------------------------------------------#

nn_pipeline  = cost_train(data, EPOCHS)

#---------------------------------------------------------------#
#                            Merging all                        #
#---------------------------------------------------------------#
# Average premium
ensemble_model = ModelEnsemble(freq_pipeline, nn_pipeline, PRIME_AVG )
# Saving the model
pickle.dump(ensemble_model, open('ensemble_model.pkl', 'wb'))
