from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import pickle

app = FastAPI(title="Customer Churn ML Model Khang HW")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return round(float(result),3)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

@app.get("/")
def read_root():
    return {"message": "Hello, the API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)