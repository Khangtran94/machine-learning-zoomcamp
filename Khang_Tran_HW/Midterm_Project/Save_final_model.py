import numpy as np 
import pandas as pd
import pickle

### relative path
path = 'EDA/train_dataset_final.csv'
df = pd.read_csv(path)

### Preprocess models:

def convert_categorize(df):
    import pandas as pd

    # Define category orders
    sleep_categories = ['< 5 hours', '5-6 hours', '7-8 hours', '> 8 hours']
    habit_categories = ['Unhealthy', 'Moderate', 'Healthy']
    age_categories = ['<20', '20-30', '30-40', '40-50', '> 50']
    age_risk = ['< 30','30+']
    risk_level = ['Low Risk','High Risk']
    financial_order = [1,2,3,4,5]
    work_study_order = [0,1,2,3,4,5]
    work_hour = [i for i in range(13)]

    # Convert columns to ordered categorical types if they exist
    df['Sleep Duration'] = pd.Categorical(df['Sleep Duration'], categories=sleep_categories, ordered=True)
    df['Dietary Habits'] = pd.Categorical(df['Dietary Habits'], categories=habit_categories, ordered=True)
    df['Age_Group'] = df['Age_Group'].str.strip()
    df['Age_Group'] = pd.Categorical(df['Age_Group'], categories=age_categories, ordered=True)
    df['Age_Risk'] = pd.Categorical(df['Age_Risk'],categories=age_risk, ordered=True)
    df['Risk_Level'] = pd.Categorical(df['Risk_Level'], categories=risk_level, ordered=True)
    df['Academic Pressure'] = pd.Categorical(df['Academic Pressure'],categories=work_study_order,ordered=True)
    df['Work Pressure'] = pd.Categorical(df['Work Pressure'], categories=work_study_order,ordered=True)
    df['Study Satisfaction'] = pd.Categorical(df['Study Satisfaction'],categories=work_study_order, ordered=True)
    df['Job Satisfaction'] = pd.Categorical(df['Job Satisfaction'], categories=work_study_order,ordered=True)
    df['Financial Stress'] = pd.Categorical(df['Financial Stress'],ordered=True,categories=financial_order)
    df['Work/Study Hours'] = pd.Categorical(df['Work/Study Hours'], ordered=True, categories=work_hour)
    
    ### Change to category types:
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype("category")
    
    return df
df = convert_categorize(df)

neg = np.sum(df['Depression'] == 0)
pos = np.sum(df['Depression'] == 1)

# print(f"Negatives (0): {neg}, Positives (1): {pos}")
scale_pos_weight = neg / pos
# print("scale_pos_weight =", round(scale_pos_weight,2))

X = df.drop('Depression',axis=1)
y = df['Depression']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y,random_state=42)
X_train.columns = X_train.columns.str.replace(" ", "_")
X_val.columns   = X_val.columns.str.replace(" ", "_")

### Training models
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

best_params = {
    'iterations': 593,                    # updated
    'depth': 5,                           # updated
    'learning_rate': 0.06156867848924621, # updated
    'l2_leaf_reg': 2.8722545484680766,    # updated
    'bagging_temperature': 0.48087662422549526, # updated
    'border_count': 93,                   # updated
    'random_seed': 42,
    'verbose': 0
}

# Identify categorical columns
categorical_cols = X_train.select_dtypes('category').columns.tolist()

cat_model = CatBoostClassifier(**best_params,allow_writing_files=False)

cat_model.fit(
    X_train, y_train,
    cat_features=categorical_cols,
    eval_set=(X_val, y_val),
    use_best_model=True)

### Save model
feature_columns = X_train.columns.tolist()  # list of all feature names
categorical_cols = X_train.select_dtypes('category').columns.tolist()

from pathlib import Path

output_path = Path(__file__).resolve().parent / "catboost_model.pkl"

# Save schema together with model
with open(output_path, 'wb') as f:
    pickle.dump({
        'model': cat_model,
        'feature_columns': feature_columns,
        'categorical_cols': categorical_cols
    }, f)

print(f"Model saved at: {output_path}")