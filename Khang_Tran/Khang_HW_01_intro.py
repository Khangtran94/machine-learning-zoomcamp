### Import pandas
import pandas as pd

### Q1 Pandas version
print('Pandas version:',pd.__version__)

### URL to the data
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"

### Read the CSV directly into a DataFrame
df = pd.read_csv(url)

### Display the first few rows
print(df.head())

### Q2. Records count
print('Records in the dataset: ',len(df))

### Q3. Distinct fuel type
print('Number of distinct value of fuel type: ',df['fuel_type'].nunique())
print('Distinct values of fuel type:',df['fuel_type'].unique())

### Q4. Columns have missing values
col_missing = []
for i in df.columns:
    if df[i].isnull().sum() > 0:
        col_missing.append(i)
        
print('Number of columns have missing values: ',len(col_missing))

### Q5. Max fuel efficiency from Asia
print('Max fuel efficiency: ',round(df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max(),2))
      
### Q6. Median value of horsepower
med_hp_before = df['horsepower'].median() ### 149
df_new = df.copy()
df_new['horsepower'] = df_new['horsepower'].fillna(df['horsepower'].value_counts().reset_index().iloc[0,0])
med_hp_after  = df_new['horsepower'].median()

if med_hp_before < med_hp_after:
    print('Yes, it increased')
elif med_hp_before == med_hp_before:
    print('No')
else:
    print('Yes, it decreased')

### Q7. Sum of weights
asia = df[df['origin'] == 'Asia'][['vehicle_weight','model_year']].head(7)
X = asia.values
