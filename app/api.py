"""An API to expose our trained pipeline for prime prediction."""
import sys
import os
from pathlib import Path
path = str(Path(os.path.split(__file__)[0]).parent)
sys.path.insert(1, path + '/src')

from fastapi import FastAPI
import pandas as pd
import joblib

model = joblib.load(path + '/src/ensemble_model.joblib')

app = FastAPI(
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
    Value: float = 10000,
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

    return prediction