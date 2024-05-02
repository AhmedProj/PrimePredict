"""An API to expose our trained pipeline for prime prediction."""


import sys
import os
from pathlib import Path
import logging
path = str(Path(os.path.split(__file__)[0]).parent)
sys.path.insert(1, path + '/src')
sys.path.insert(2, path)

from fastapi import FastAPI
import pandas as pd
import joblib

from contextlib import asynccontextmanager

from app.utils import get_model, ModelEnsemble

logging.basicConfig(filename="log_file.log", 
					format="%(asctime)s - %(levelname)s - %(message)s", 
					filemode='w') 

# creating an object
logger=logging.getLogger() 

logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_freq
    global model_reg
    global model

    model_freq_name: str = os.getenv("MLFLOW_MODEL_FREQ_NAME")
    model_freq_version: str = os.getenv("MLFLOW_MODEL_FREQ_VERSION")
    model_reg_name: str = os.getenv("MLFLOW_MODEL_REG_NAME")
    model_reg_version: str = os.getenv("MLFLOW_MODEL_REG_VERSION")
    # Load the ML model
    model_freq = get_model(model_freq_name, model_freq_version)
    model_reg = get_model(model_reg_name, model_reg_version)
    model = ModelEnsemble(model_freq, model_reg)
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Prédiction prime",
    description="test")


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction de prime",
        "Model_name": 'Prime ML',
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    Type: str = "A",
    Occupation: str = "Employed",
    Age: float = 30,
    Group1: int = 3,
    Bonus: int = 23,
    Poldur: int = 45,
    Value: float = 1000.0,
    Adind: int = 1,
    Density: float = 100.0,
    Exppdays: float = 365
) -> float:
    """
    Predict function of the API.
    """

    df = pd.DataFrame(
        {
            "Type": [Type], 
            "Occupation": [Occupation],
            "Age": [Age],
            "Group1": [Group1],
            "Bonus": [Bonus],
            "Poldur": [Poldur],
            "Value": [Value],
            "Adind": [Adind],
            "Density": [Density],
            "Exppdays": [Exppdays]
        }
    )

    prediction = model.transform(df)
    print(f"Response: {prediction}")
    logger.info(f"Response: {prediction}")
    return prediction