## ML Zoomcamp 2025 Khang's Midterm Project
# MENTAL HEALTH PREDICTION
<img width="444" height="433" alt="Screenshot 2025-11-17 at 23 10 42" src="https://github.com/user-attachments/assets/9f18ec0c-4945-4161-af38-4e3d7fcd8160" />

## Problem Description:
* Business Perspective:
Mental health risk prediction helps organizations and healthcare providers identify at-risk individuals early, enabling targeted interventions, cost-effective support, and better resource allocation.

* Data Science Perspective:
The dataset is a synthetic mental health survey. The task is to predict a binary outcome (e.g., depression risk) from features like demographics, stress, and family history. Challenges include categorical data, numerical data, possible class imbalance

* Approach:
1. Perform EDA and preprocess categorical features.
2. Handle missing data and class imbalance.
3. Train ML models (XGBoost, LightGBM, CatBoost) with cross-validation.
4. Interpret results using feature importance/SHAP to provide actionable insights.
5. Compare between those metrics: F1-score, ROC-AUC, confusion matrix, classification report then select the final model
 
## How to use

Make sure you have Python 3.10+ installed. Install dependencies:

```bash
pip install -r requirements.txt
```
The project includes a simple web API for predictions:

```bash
cd Web_Service
python Web_Model.py
```
The service uses FastAPI for predictions (or Pydantic validation if using Web_Model_Pydantic.py).
You can access it at http://127.0.0.1:8000 (or the port you specify).

Once the web service is running, you can test a prediction:

```bash
# Example using curl
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"feature1": value1, "feature2": value2, "feature3": value3}'
Replace feature1, feature2, etc., with your input features.
```
### Docker App
Open Docker Desktop then Docker Hub => find **khangtranvn/mental_health_prediction_app**

![alt text](<Screenshot 2025-11-18 at 00.50.08.png>)

## Dataset Overview:
The data comes from the [Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11/data) Kaggle competition.

Overview of the dataset. 140700 rows and 20 columns
<img width="1746" height="330" alt="Screenshot 2025-11-17 at 23 16 20" src="https://github.com/user-attachments/assets/6a8fe7b2-77e1-45a6-ab9b-cf5c6c3571ec" />

We can see many missing values:

<img width="1205" height="644" alt="Screenshot 2025-11-17 at 23 17 10" src="https://github.com/user-attachments/assets/5e516c7c-610a-4b87-aadd-0f63db926e43" />

<img width="597" height="449" alt="Screenshot 2025-11-17 at 23 17 48" src="https://github.com/user-attachments/assets/24cbf857-1f2d-4cba-a754-7f1dccb00693" />

### Cleaning data: 
Please refer to the notebook [Cleaning_data](https://github.com/Khangtran94/machine-learning-zoomcamp/blob/ML_Homework_Khang/Khang_Tran_HW/Midterm_Project/Notebooks/2_EDA.ipynb)

## EDA: 
Please refer to the notebook [EDA](https://github.com/Khangtran94/machine-learning-zoomcamp/blob/ML_Homework_Khang/Khang_Tran_HW/Midterm_Project/Notebooks/2_EDA.ipynb) for more information.
* Mutual Information:
  <img width="990" height="819" alt="Screenshot 2025-11-17 at 23 20 39" src="https://github.com/user-attachments/assets/1c030a9c-edfd-4746-af69-15fd40708904" />
* Correlation:
  <img width="908" height="818" alt="Screenshot 2025-11-17 at 23 21 12" src="https://github.com/user-attachments/assets/69eacea9-f7e3-4064-9e6d-98fa94d39fd3" />
* Suicidal thoughts and financial difficulties were key factors influencing the target variable.
<img width="822" height="402" alt="Screenshot 2025-11-17 at 23 21 31" src="https://github.com/user-attachments/assets/a993ed07-aab2-44fe-b620-a179ae321e88" />

* When we combine those two together
<img width="672" height="399" alt="Screenshot 2025-11-17 at 23 23 16" src="https://github.com/user-attachments/assets/2897c962-1f5a-491a-a71d-97b99a0b9273" />

* Depression varies across different age groups.
<img width="819" height="335" alt="Screenshot 2025-11-17 at 23 24 18" src="https://github.com/user-attachments/assets/07808aa4-47ce-4fa1-8884-00301111a23d" />

* Most of the data comes from individuals over 30, but many under 30 report experiencing depression.
<img width="592" height="426" alt="Screenshot 2025-11-17 at 23 25 06" src="https://github.com/user-attachments/assets/b0a221c4-a779-4424-9b0b-a14e3b5d479c" />

* Depression by Group Age and Suicidal Thoughts:
  <img width="985" height="580" alt="Screenshot 2025-11-17 at 23 27 13" src="https://github.com/user-attachments/assets/5ddc9024-9673-475f-be04-642a21594bb5" />

* Depression by Group Age and Financial Stress:
<img width="984" height="583" alt="Screenshot 2025-11-17 at 23 27 23" src="https://github.com/user-attachments/assets/f79852d7-e2d5-484f-bbba-d88422356de1" />

## Model Training:
I trained 3 tree-based models: **XGBoost**, **LightGBM**, **Catboost**. 
I also used **Optuna** for fine-tuned hyperparameters. Please refer to notebook [Optuna Training](https://github.com/Khangtran94/machine-learning-zoomcamp/blob/ML_Homework_Khang/Khang_Tran_HW/Midterm_Project/Notebooks/4_Optuna.ipynb)

Compare performance between 3 models:
* ROC-AUC curve:
  <img width="816" height="540" alt="Screenshot 2025-11-17 at 23 32 44" src="https://github.com/user-attachments/assets/0dc4967a-23fa-487a-9426-5ffa8f1eb50a" />

* Confusion Matrix:
<img width="1459" height="434" alt="Screenshot 2025-11-17 at 23 33 08" src="https://github.com/user-attachments/assets/175055fe-0c17-44ae-beb8-e16da01d27fd" />

* Feature Importance:
<img width="1456" height="431" alt="Screenshot 2025-11-17 at 23 33 23" src="https://github.com/user-attachments/assets/ebecad5c-6128-47b4-afca-c73e853eb79b" />

* Classification Report:
<img width="1463" height="404" alt="Screenshot 2025-11-17 at 23 33 51" src="https://github.com/user-attachments/assets/57499e7f-2e5b-4012-9bc1-3aa218eb99bd" />

Based on the comparision between those metrics, I decided to choose **CATBOOST** model as my final model

## Save model as pickle file:
<img width="1008" height="29" alt="Screenshot 2025-11-17 at 23 50 25" src="https://github.com/user-attachments/assets/482f1269-9094-4e4a-a0a5-b971e57f873d" />

I save model as catboost_model.pkl

## Model Deployment:
* Option 1: via FastAPI UI.
  * Run predict with Example False Predict Depresion
<img width="732" height="867" alt="Screenshot 2025-11-17 at 23 55 14" src="https://github.com/user-attachments/assets/1e93f744-dc57-4866-94de-54dea47178b7" />

  * One example with Predict = True
 
  <img width="731" height="861" alt="Screenshot 2025-11-17 at 23 55 48" src="https://github.com/user-attachments/assets/ba569c12-9a7a-42a5-9614-7a8ffe907553" />

* Option 2: Edit the JSON file in **Load_final_model.py** then run via Terminal

 * One example Predict = False

<img width="371" height="130" alt="Screenshot 2025-11-17 at 23 57 18" src="https://github.com/user-attachments/assets/f500deda-8830-4c0b-b259-2ad05df35dea" />

  
<img width="1129" height="42" alt="Screenshot 2025-11-17 at 23 57 33" src="https://github.com/user-attachments/assets/4fab0f35-4699-4839-bb6b-355d61e6dba2" />


   * One example Predict = True

<img width="394" height="133" alt="Screenshot 2025-11-17 at 23 58 30" src="https://github.com/user-attachments/assets/a5984f37-4a7d-455a-a83f-ca5706ddc89d" />


<img width="1121" height="43" alt="Screenshot 2025-11-17 at 23 58 37" src="https://github.com/user-attachments/assets/4a4db0ea-7306-49e9-a81f-5652257502a9" />

## Dependency and Environment Management:

I created [requirements.txt](https://github.com/Khangtran94/machine-learning-zoomcamp/blob/ML_Homework_Khang/Khang_Tran_HW/Midterm_Project/requirements.txt)

## Containerization:
after create [Dockerfile](https://github.com/Khangtran94/machine-learning-zoomcamp/blob/ML_Homework_Khang/Khang_Tran_HW/Midterm_Project/Dockerfile)

1. Build Docker images **mental_health_prediction_app** in github codespace
   <img width="1428" height="272" alt="Screenshot 2025-11-18 at 00 06 34" src="https://github.com/user-attachments/assets/02646372-4440-4577-9172-20826658ddea" />

2. Check run Docker app inside Github Codespace:
   <img width="1186" height="144" alt="Screenshot 2025-11-18 at 00 12 32" src="https://github.com/user-attachments/assets/c9bd47e3-b868-40bc-807e-ea11a40a6e5d" />

<img width="1443" height="626" alt="Screenshot 2025-11-18 at 00 12 44" src="https://github.com/user-attachments/assets/a691bfd4-db17-42e4-a80d-d77dc8dc947c" />

3. Push Docker Images to Docker Hub
<img width="1244" height="179" alt="Screenshot 2025-11-18 at 00 17 29" src="https://github.com/user-attachments/assets/46a31757-618d-4146-baf1-0ae2ef062f03" />

4. Check in Docker Hub
   Follow this link [DockerHub](https://hub.docker.com/repository/docker/khangtranvn/mental_health_prediction_app/general)
<img width="1434" height="783" alt="Screenshot 2025-11-18 at 00 17 57" src="https://github.com/user-attachments/assets/5ca0a647-74fa-402b-9ac0-c9aa0731b844" />
   
5. Open Docker Desktop, pull images then run
   <img width="739" height="460" alt="Screenshot 2025-11-18 at 00 28 22" src="https://github.com/user-attachments/assets/f317ce6a-6d93-46c0-a540-e8056f74b215" />

FINALLY I CAN BUILD MY MODEL ON DOCKER.
Thank you teacher and everyone so much

## Cloud Deployment:
not deploy to Cloud yet T.T I am still learning about it
