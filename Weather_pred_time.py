#!/usr/bin/env python
# coding: utf-8

# In[348]:


#Weather Prediction


# In[349]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[350]:


#reading dataset
data = pd.read_csv(r"C:\Users\user\Desktop\IOT\weather.csv")


# In[351]:


df = pd.DataFrame(data)


# In[352]:


#dropping nan values
df_dropped = df.dropna()
df_dropped


# In[353]:


#converting the type of read_dt (object) to datetime64
df_dropped['read_dt'] = pd.to_datetime(df_dropped['read_dt'])

# Convert the date into a number (of days since some point)
fromDate = min(df_dropped['read_dt'])
df_dropped['timedelta'] = (df_dropped['read_dt'] - fromDate).dt.days.astype(int)
print(df_dropped[['read_dt', 'timedelta']].head())

#dropping the read_dt column which will not be used anymore ( the column timedelta will be used instead )
df_dropped.drop('read_dt', axis = 1, inplace = True)


# In[354]:


#converting the read_tm column ( object ) to seconds which will be used as time for the date
df_dropped['seconds'] = df_dropped.read_tm.apply(pd.to_timedelta).apply(lambda x: x.total_seconds())

#dropping the read_tm column which will not be used anymore ( the column seconds will be used instead )
df_dropped.drop('read_tm', axis = 1, inplace = True)


# In[355]:


#changing categorical data features ( Condition , wind_direciton) to numerical features that the machine can understand it ( one hot encoding )
df_dropped  = pd.get_dummies(df_dropped,prefix=['condition','wind_direction'],columns=['condition','wind_direction'])

#Dataset which will be used after data preparation
df_dropped


# In[356]:


#Splitting Features and lables ( we will try to predict the 'air_quality_health_index')
from sklearn.model_selection import train_test_split
X = df_dropped.drop('air_quality_health_index', axis = 1)
y = np.array(df_dropped['air_quality_health_index'])


# In[381]:


#Splitting data to test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[382]:


#preparing random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[383]:


#training model
rf.fit(X_train, y_train);


# In[384]:


#making predictions on test set
predictions = rf.predict(X_test)


# In[385]:


#Calculating errors
errors = abs(predictions - y_test)


# In[386]:


#Mean absolute error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[387]:


#Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)


# In[388]:


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




