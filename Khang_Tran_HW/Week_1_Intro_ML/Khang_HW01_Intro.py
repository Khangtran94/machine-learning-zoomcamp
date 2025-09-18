import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'
df = pd.read_csv(url)
print(df)

### Q1. Pandas version
print('Pandas version:',pd.__version__)

### Q2. Records count
print('Total rows of the cars dataset: ',len(df))

### Q3. Fuel types
print('Unique values of fuel type: ',df['fuel_type'].unique())
print()
print('Number of unique fuel types:',df['fuel_type'].nunique())

### Q4. Missing values
col_missing = 0
for i in df.columns:
    if df[i].isnull().sum() > 0:
        col_missing += 1
print('Number of columns have missing values:',col_missing)

### Q5. Max fuel efficiency
print('Maximum fuel efficiency of cars from Asia:',round(df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max(),2))

### Q6. Median value of horsepower
med_hp_before = df['horsepower'].median()
df_new = df.copy()
df_new['horsepower'] = df_new['horsepower'].fillna(df['horsepower'].value_counts().reset_index().iloc[0,0])
med_hp_after = df_new['horsepower'].median()
if med_hp_before < med_hp_after:
    print('Yes, it INCREASED')
elif med_hp_before > med_hp_after:
    print('Yes, it DECREASED')
else:
    print('NO')


### Q7. Sum of weights
X = df[df['origin'] == 'Asia'][['vehicle_weight','model_year']].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print('Sum of weights:',round(sum(w),2))