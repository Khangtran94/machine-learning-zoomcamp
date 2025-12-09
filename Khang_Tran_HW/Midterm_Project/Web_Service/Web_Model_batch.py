from fastapi import FastAPI
from typing import Dict, Any, List, Union
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
    json_input: Union[Dict[str, Any], List[Dict[str, Any]]],
    feature_columns: list,
    categorical_cols: list,
    fill_numeric: float = 0.0,) -> pd.DataFrame:
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


def predict_probabilities(input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    df = prepare_input(input_data, feature_columns, categorical_cols)
    probs = loaded_model.predict_proba(df)[:, 1]
    results = []
    for prob in probs:
        results.append({
            "depression_probability": round(float(prob), 3),
            "depression": bool(prob >= 0.5)
        })
    return results


@app.post("/predict")
def predict(customer: Union[Dict[str, Any], List[Dict[str, Any]]]):
    return predict_probabilities(customer)

@app.get("/")
def read_root():
    return {"message": "Hello! The Mental Health Prediction API is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
