from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
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


# ------------------------------
# Pydantic Models for Input
# ------------------------------
class CustomerInput(BaseModel):
    Age: Optional[int] = Field(None, description="Age of the individual")
    suicidal_thoughts: Optional[str] = Field(
        None,
        alias="Have_you_ever_had_suicidal_thoughts_?",
        description="Yes/No for suicidal thoughts"
    )
    Work_Pressure: Optional[int] = Field(None, description="Pressure score")
    Financial_Stress: Optional[int] = Field(None, description="Financial stress score")
    Job_Satisfaction: Optional[int] = Field(None, description="Job/Study satisfaction score")
    Work_Study_Hour: Optional[int] = Field(
    None,
    alias="Work_Study_Hours",
    description="Work/Study Hours")
    # Add other features as needed

class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput]


# ------------------------------
# Helper Functions
# ------------------------------
def prepare_input(
    json_input: List[dict],
    feature_columns: list,
    categorical_cols: list,
    fill_numeric: float = 0.0,
) -> pd.DataFrame:
    """
    Converts JSON input to a DataFrame, fills missing columns,
    ensures correct order, and handles categorical/numeric columns.
    """
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

def predict_probabilities(input_data: List[dict]) -> List[dict]:
    """
    Predicts probabilities for single or multiple inputs.
    Returns a list of dictionaries with probability and boolean label.
    """
    df = prepare_input(input_data, feature_columns, categorical_cols)
    probs = loaded_model.predict_proba(df)[:, 1]
    results = []
    for prob in probs:
        results.append({
            "depression_probability": round(float(prob), 3),
            "depression": bool(prob >= 0.5)
        })
    return results


# ------------------------------
# API Endpoints
# ------------------------------
@app.post("/predict")
def predict(batch_input: BatchCustomerInput):
    """
    Accepts a list of customers.
    Returns probability of depression for each entry.
    """
    input_dicts = [customer.dict() for customer in batch_input.customers]
    return predict_probabilities(input_dicts)


@app.get("/")
def read_root():
    return {"message": "Hello! The Mental Health Prediction API is running."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
