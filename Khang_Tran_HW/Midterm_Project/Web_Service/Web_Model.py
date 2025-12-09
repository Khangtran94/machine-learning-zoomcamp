from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Mental Health Prediction API")

model_path = Path(__file__).resolve().parent.parent / "catboost_model.pkl"

print("Loading model from:", model_path)

with open(model_path, "rb") as f:
    saved = pickle.load(f)

loaded_model = saved["model"]
feature_columns = saved["feature_columns"]
categorical_cols = saved["categorical_cols"]


def prepare_input(
    json_input: Dict[str, Any],
    feature_columns: list,
    categorical_cols: list,
    fill_numeric: float = 0.0,
) -> pd.DataFrame:
    """
    Converts JSON input into a DataFrame suitable for the model.
    Fills missing columns with default values.
    """
    # Convert JSON to DataFrame
    if isinstance(json_input, dict):
        df = pd.DataFrame([json_input])
    else:
        df = pd.DataFrame(json_input)

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            if col in categorical_cols:
                df[col] = "Unknown"
            else:
                df[col] = fill_numeric

    # Ensure correct column order
    df = df[feature_columns]
    return df


def predict_single(customer: Dict[str, Any]) -> float:
    """
    Predicts the probability of depression for a single input.
    """
    input_df = prepare_input(customer, feature_columns, categorical_cols)
    prob = loaded_model.predict_proba(input_df)[:, 1][0]
    return round(float(prob), 3)

# ----------------------
# API Endpoints
# ----------------------
@app.post("/predict")
def predict(customer: Dict[str, Any]):
    """
    POST endpoint for predicting depression probability.
    Expects JSON with feature values.
    """
    prob = predict_single(customer)
    return {
        "depression_probability": prob,
        "depression": bool(prob >= 0.5),
    }

@app.get("/")
def read_root():
    """
    Root endpoint to check if API is running.
    """
    return {"message": "Hello! The Mental Health Prediction API is running."}

# Optional GET route to remind users to use POST
@app.get("/predict")
def predict_get():
    return {"message": "Please use POST with JSON data to get predictions."}

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)