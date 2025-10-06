import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)
df = df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]
df

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(x='fuel_efficiency_mpg', data=df)
plt.show()

### Q1. Column have missing values:
col_missing = []
for i in df.columns:
    if df[i].isnull().sum() > 0:
        col_missing.append(i)
print('Column have missing values:',col_missing)


### Q2. Median for *horsepower*:
print('Median for horsepower:',df['horsepower'].median())

#### Prepare and split the dataset (same code in lecture)
### Train / val / test distribution:
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

n_train, n_val, n_test

### Random dataset
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

### Subset dataframe
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

### Check number of rows
assert (len(df_train), len(df_val), len(df_test)) == (n_train, n_val, n_test)

### Fill with 0:
train_fill_0 = df_train.fillna(0)

### Fill with mean:
train_fill_mean = df_train.fillna(df['horsepower'].mean())

### Q3. RMSE baseline Linear Regression without Regularization:
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

X_val = df_val[['engine_displacement','horsepower','vehicle_weight','model_year']].fillna(0).values
y_val = df_val['fuel_efficiency_mpg'].values

### For fill with 0
X_train_fill_0 = train_fill_0[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_train_fill_0 = train_fill_0['fuel_efficiency_mpg'].values
w_0_0, w_0 = train_linear_regression(X_train_fill_0, y_train_fill_0)

y_pred_fill_0 = w_0_0 + X_val.dot(w_0)
rmse_fill_0 = rmse(y_val, y_pred_fill_0)
rmse_fill_0 = float(round(rmse_fill_0,2))
rmse_fill_0

### For fill with mean:
X_train_fill_mean = train_fill_mean[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_train_fill_mean = train_fill_mean['fuel_efficiency_mpg'].values
w_0_mean, w_mean = train_linear_regression(X_train_fill_mean, y_train_fill_mean)

y_pred_fill_mean = w_0_mean + X_val.dot(w_mean)
rmse_fill_mean = rmse(y_val, y_pred_fill_mean)
rmse_fill_mean = float(round(rmse_fill_mean,2))
rmse_fill_mean

if rmse_fill_0 < rmse_fill_mean:
    print('Filling missing values with 0 is better. RMSE:')
    print(rmse_fill_0)
elif rmse_fill_mean == rmse_fill_0:
    print('Both are equally good. RMSE:')
    print(rmse_fill_0)
else:
    print('Filling missing values with MEAN is better. RMSE:')
    print(rmse_fill_mean)   

### Q4. r give the best RMSE with Regularized Linear Regression:
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

r_list = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_r_dict = {}
for r in r_list:
    w_0_r, w_r = train_linear_regression_reg(X_train_fill_0, y_train_fill_0, r)
    y_pred_r = w_0_r + X_val.dot(w_r)
    rmse_r = float(round(rmse(y_val, y_pred_r),4))
    rmse_r_dict[r] = rmse_r

rmse_r_dict

### Q5. Value of std of RMSE between seeds:
seeds = list(range(10))
rmse_scores = []
for seed in seeds:
    idx = np.arange(len(df))
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    train_fill_0 = df_train.fillna(0)

    X_val = df_val[['engine_displacement','horsepower','vehicle_weight','model_year']].fillna(0).values
    y_val = df_val['fuel_efficiency_mpg'].values

    X_train_fill_0 = train_fill_0[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y_train_fill_0 = train_fill_0['fuel_efficiency_mpg'].values
    w_0_0, w_0 = train_linear_regression(X_train_fill_0, y_train_fill_0)

    y_pred_fill_0 = w_0_0 + X_val.dot(w_0)
    rmse_fill_0 = rmse(y_val, y_pred_fill_0)
    rmse_fill_0 = float(round(rmse_fill_0,6))
    print(f'Seed: {seed}, RMSE: {rmse_fill_0}')
    rmse_scores.append(rmse_fill_0)
print()
print('RMSE scores for all seeds:',rmse_scores)
print('Std of RMSE between different seeds:',round(np.std(rmse_scores),3))

### Q6. Best RMSE on test dataset:
idx = np.arange(len(df))
np.random.seed(9)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_trainval = pd.concat([df_train, df_val], ignore_index=True)
df_trainval = df_trainval.fillna(0)

X_trainval = df_trainval[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_trainval = df_trainval['fuel_efficiency_mpg'].values

X_test = df_test[['engine_displacement','horsepower','vehicle_weight','model_year']].fillna(0).values
y_test = df_test['fuel_efficiency_mpg'].values

w_0_final, w_final = train_linear_regression_reg(X_trainval, y_trainval,r=0.001)
y_pred_test = w_0_final + X_test.dot(w_final)
rmse_test = float(round(rmse(y_test, y_pred_test),3))
rmse_test
