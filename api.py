"""An API to expose our trained pipeline for prime prediction."""

from fastapi import FastAPI
import pandas as pd
import joblib


model = joblib.load('src/ensemble_model.joblib')

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
    Type: str = "test",
    Occupation: str = "test2",
    Age: float = 30,
    Group1: int = 3,
    Bonus: int = 23,
    Poldur: int = 45,
    Value: float = 34,
    Adind: int = 34,
    Density: float = 20,
    Exppdays: float = 23
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