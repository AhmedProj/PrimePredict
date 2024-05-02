import sys
import os
from pathlib import Path

path = Path(os.path.split(__file__)[0])
sys.path.insert(1, str(path.parent))

from itertools import product
import pandas as pd
import numpy as np
import yaml
from model.train_models import frequency_train, cost_train
from pipeline.preprocessing import preprocessing
import mlflow


def log_to_mlflow(
    mlflow_experiment_name, remote_server_uri, params="params.yaml", reg=True, freq=True
):

    # Fichier hyperparametres
    with open(params, "r") as file:
        hyperparams = yaml.safe_load(file)

    data = pd.read_csv(str(path.parent) + "/training.csv", sep=";")
    data = preprocessing(data)

    PRIME_AVG = sum(np.expm1(data["total_cost"])) / len(data["total_cost"])

    combinations_1 = list(
        product(
            *(hyperparams["freq"][name] for name in list(hyperparams["freq"].keys()))
        )
    )
    combinations_2 = list(
        product(*(hyperparams["reg"][name] for name in list(hyperparams["reg"].keys())))
    )

    # Set up MLFlow context
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    if freq:

        for idx, e in enumerate(combinations_1):
            # For each hyperparameter combination we trained the model with, we log a run in MLflow
            run_name = f"run_freq {idx}"
            with mlflow.start_run(run_name=run_name):
                freq_pipeline = frequency_train(
                    df=data, kernel=e[0], degree=e[1], class_weight=e[2]
                )

                mlflow.log_params(
                    {"kernel": e[0], "degree": e[1], "class_weight": e[2]}
                )

                # Log fit metrics a ajouter

                # Log model as an artifact
                mlflow.sklearn.log_model(freq_pipeline, "freq_model")
    if reg:
        for idx, e in enumerate(combinations_2):
            # For each hyperparameter combination we trained the model with, we log a run in MLflow
            run_name = f"run_reg {idx}"
            with mlflow.start_run(run_name=run_name):
                reg_pipeline = cost_train(
                    df=data,
                    n_estimators=e[0],
                    max_depth=e[1],
                    test_path="test_cost.csv",
                )

                mlflow.log_params({"n_estimators": e[0], "max_depth": e[1]})

                # Log fit metrics a ajouter

                # Log model as an artifact
                mlflow.sklearn.log_model(reg_pipeline, "reg_model")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        experiment = str(sys.argv[1])

    log_to_mlflow(
        mlflow_experiment_name=experiment,
        remote_server_uri="https://user-ahmed-mlflow.user.lab.sspcloud.fr",
        params="params.yaml",
        reg=True,
        freq=True,
    )
