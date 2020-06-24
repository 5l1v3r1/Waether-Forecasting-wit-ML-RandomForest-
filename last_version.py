#!/usr/bin/env python
# coding: utf-8

# In[559]:


#Weather Prediction


# In[560]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[561]:


#reading dataset
data = pd.read_csv(r"C:\Users\user\Desktop\IOT\weather.csv")


# In[562]:


#creating dataframe
df = pd.DataFrame(data)


# In[563]:


#dropping nan values
df_dropped = df.dropna()
df_dropped


# In[564]:


#dates = np.array(df_dropped.read_dt)
#len(dates)


# In[565]:


#from datetime import datetime
#month_day = []

#for i in range(0,len(dates)):
#    your_datetime = dates[i]
#    month_day.append(datetime.strptime(your_datetime , "%Y-%m-%d").strftime('%m-%d'))
    
    
#days =pd.DataFrame(month_day)
#days


# In[566]:


#converting the type of read_dt (object) to datetime64
#df_dropped['read_dt'] = pd.to_datetime(df_dropped['read_dt'])

#Convert the date into a number (of days since some point)
#fromDate = min(df_dropped['read_dt'])
#df_dropped['timedelta'] = (df_dropped['read_dt'] - fromDate).dt.days.astype(int)
#print(df_dropped[['read_dt', 'timedelta']].head())

#dropping the read_dt column which will not be used anymore ( the column timedelta will be used instead )
#df_dropped.drop('read_dt', axis = 1, inplace = True)
#df_dropped.timedelta


# In[567]:


#converting the read_tm column ( object ) to seconds which will be used as time for the date
#df_dropped['secods'] = df_dropped.read_tm.apply(pd.to_timedelta).apply(lambda x: x.total_seconds())

#dropping the read_tm column which will not be used anymore ( the column seconds will be used instead )
#df_dropped.drop('read_tm', axis = 1, inplace = True)


# In[568]:


#changing categorical data features ( Condition , wind_direciton) to numerical features that the machine can understand it ( one hot encoding )
df_dropped  = pd.get_dummies(df_dropped,prefix=['read_dt','read_time','condition','wind_direction'],columns=['read_dt','read_tm','condition','wind_direction'])

#Dataset which will be used after data preparation
df_dropped


# In[569]:


#Splitting Features and lables ( we will try to predict the 'air_quality_health_index')
from sklearn.model_selection import train_test_split
X = df_dropped.drop('air_quality_health_index', axis = 1)
y = np.array(df_dropped['air_quality_health_index'])
X = X[:-1]
X = X.drop(X.index[0])
X


# In[570]:


from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit()
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[480]:


#train_pct_index = int(0.80 * len(X))
#X_train, X_test = X[:train_pct_index], X[train_pct_index:]
#y_train, y_test = y[:train_pct_index], y[train_pct_index:]


# In[481]:


#Splitting data to test and train
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 24)


# In[482]:


#preparing random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[483]:


#training model
rf.fit(X_train, y_train);


# In[ ]:


#making predictions on test set
predictions = rf.predict(X_test)
predictions = predictions.round()
#sort ettik
#predictions = np.sort(predictions)
#y_test = np.sort(y_test)


# In[ ]:


#Calculating errors
errors = abs(predictions - y_test)


# In[ ]:


#Mean absolute error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


#Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)


# In[ ]:


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(len(predictions)),predictions,color  = "blue")
plt.plot(range(len(y_test)),y_test,color  = "red", alpha = 0.4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




