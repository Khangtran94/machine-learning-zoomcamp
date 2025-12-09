import pickle
import pandas as pd
from pathlib import Path

output_path = Path(__file__).resolve().parent / "catboost_model.pkl"

### Load the model:
with open(output_path, 'rb') as f:
    saved = pickle.load(f)

loaded_model = saved['model']
feature_columns = saved['feature_columns']
categorical_cols = saved['categorical_cols']

def prepare_input(json_input, feature_columns, categorical_cols, fill_numeric=0):
    # Convert JSON to DataFrame
    if isinstance(json_input, dict):
        df = pd.DataFrame([json_input])
    else:
        df = pd.DataFrame(json_input)

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            if col in categorical_cols:
                df[col] = 'Unknown'  # or first category if known
            else:
                df[col] = fill_numeric

    # Ensure correct column order
    df = df[feature_columns]
    return df

### test 
json_input = {
    "Age": "20",
    "Have_you_ever_had_suicidal_thoughts_?": "Yes",
    "Work_Pressure" : "5",
    "Financial_Stress": "5",
    "Job_Satisfaction": "4",
    "Work/Study_Hours":"10"}

input_df = prepare_input(json_input, feature_columns, categorical_cols)
y_pred_prob = loaded_model.predict_proba(input_df)[:, 1]
print(json_input)
print(f"Predicted probability of Depression = 1: {y_pred_prob[0]:.3f}")