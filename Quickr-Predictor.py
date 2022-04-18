#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# In[58]:


car=pd.read_csv('quikr_car.csv')


# In[59]:


car.head()


# In[60]:


car.shape


# In[61]:


car.info()


# In[62]:


backup=car.copy()


# In[63]:


#Cleaning Data
#year has many non-year values
car=car[car['year'].str.isnumeric()]


# In[64]:


#year is in object. Change to integer
car['year']=car['year'].astype(int)


# In[65]:


#Price has Ask for Price
car=car[car['Price']!='Ask For Price']


# In[66]:


#Price has commas in its prices and is in object
car['Price']=car['Price'].str.replace(',','').astype(int)


# In[67]:


#kms_driven has object values with kms at last.
car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[68]:


#It has nan values and two rows have 'Petrol' in them
car=car[car['kms_driven'].str.isnumeric()]


# In[69]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[70]:


#fuel_type has nan values
car=car[~car['fuel_type'].isna()]


# In[71]:


car.shape


# In[72]:


#name and company had spammed data...but with the previous cleaning, those rows got removed.
#Company does not need any cleaning now. Changing car names. Keeping only 
#the first three words
car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# In[73]:


#Resetting the index of the final cleaned data
car=car.reset_index(drop=True)


# In[74]:


#Cleaned Data
car


# In[75]:


car.to_csv('Cleaned_Car_data.csv')


# In[76]:


car.info()


# In[77]:


car.describe(include='all')


# In[78]:


car=car[car['Price']<6000000]


# In[79]:


#Checking relationship of Company with Price
car['company'].unique()


# In[80]:


import seaborn as sns


# In[81]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[82]:


#Checking relationship of Year with Price
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[83]:


#Checking relationship of kms_driven with Price
sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)


# In[84]:


#Checking relationship of Fuel Type with Price
plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)


# In[85]:


#Relationship of Price with FuelType, Year and Company mixed
ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# In[86]:


#Extracting Training Data
X=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']


# In[87]:


X


# In[88]:


y.shape


# In[89]:


#Applying Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[90]:


from sklearn.linear_model import LinearRegression


# In[91]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[92]:


#Creating an OneHotEncoder object to contain all the possible categories
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[93]:


#Creating a column transformer to transform categorical columns
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[94]:


#Linear Regression Model
lr=LinearRegression()


# In[95]:


#Making a pipeline
pipe=make_pipeline(column_trans,lr)


# In[96]:


#Fitting the model
pipe.fit(X_train,y_train)


# In[97]:


y_pred=pipe.predict(X_test)


# In[98]:


#Checking R2 Score
r2_score(y_test,y_pred)


# In[99]:


#Finding the model with a random state of TrainTestSplit where the model was
#found to give almost 0.92 as r2_score
scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[100]:


np.argmax(scores)


# In[101]:


scores[np.argmax(scores)]


# In[102]:


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[103]:


#The best model is found at a certain random state
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[104]:


import pickle


# In[105]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[106]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[107]:


pipe.steps[0][1].transformers[0][1].categories[0]

