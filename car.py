import pandas as pd
import numpy as np
df=pd.read_csv('quikr_car.csv')
df.head()
df.shape
df.info()
df['year'].unique()
df['Price'].unique ()
df['kms_driven'].unique()
df['fuel_type'].unique()
backup=df.copy()
df=df[df['year'].str.isnumeric()]
df['year']=df['year'].astype(int)
df['year'].info()
df=df[df['Price']!='Ask For Price']
df['Price']=df['Price'].astype(str).str.replace(',','').astype(int)
df['Price'].info()
df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
df = df.dropna(subset=['kms_driven'])
df['kms_driven'] = df['kms_driven'].astype(int)
df['kms_driven'].info()
df=df[~df['fuel_type'].isna()]
df['name']=df['name'].str.split(' ').str.slice(0,3).str.join(' ')
df=df.reset_index(drop=True)
df.describe()
df=df[df['Price']<6e6]
df.to_csv('Cleaned_Car_data.csv')
import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()
sns.relplot(x='kms_driven',y='Price',data=df,height=7,aspect=1.5)
plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=df)
ax=sns.relplot(x='company',y='Price',data=df,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')
x=df.drop(columns='Price')
y=df['Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
one=OneHotEncoder()
one.fit(x[['name','company','fuel_type']])
one.categories_
column_trans=make_column_transformer((OneHotEncoder(categories=one.categories_),['name','company','fuel_type']),remainder='passthrough')
l=LinearRegression()
pipe=make_pipeline(column_trans,l)
pipe.fit(x_train,y_train)
y_pred= pipe.predict(x_test)
y_pred
r2_score(y_test,y_pred)
scores=[]
for i in range(1000):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
  l=LinearRegression()
  pipe=make_pipeline(column_trans,l)
  pipe.fit(x_train,y_train)
  y_pred=pipe.predict(x_test)
  scores.append(r2_score(y_test,y_pred))
np.argmax(scores)
scores[np.argmax(scores)]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
l=LinearRegression()
pipe=make_pipeline(column_trans,l)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
r2_score(y_test,y_pred)
import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))





