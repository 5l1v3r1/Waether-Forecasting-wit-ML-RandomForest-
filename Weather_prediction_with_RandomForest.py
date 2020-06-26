#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Weather Prediction


# In[45]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[46]:


#reading dataset
data = pd.read_csv(r"C:\Users\user\Desktop\IOT\weather.csv")


# In[47]:


#creating dataframe
df = pd.DataFrame(data)


# In[48]:


#dropping nan values
df_dropped = df.dropna()
df_dropped


# In[49]:


#changing categorical data features ( Condition , wind_direciton) to numerical features that the machine can understand it ( one hot encoding )
df_dropped  = pd.get_dummies(df_dropped,prefix=['read_dt','read_time','condition','wind_direction'],columns=['read_dt','read_tm','condition','wind_direction'])

#Dataset which will be used after data preparation
df_dropped


# In[50]:


#Splitting Features and lables ( we will try to predict the 'air_quality_health_index')
from sklearn.model_selection import train_test_split
X = df_dropped.drop('air_quality_health_index', axis = 1)
y = np.array(df_dropped['air_quality_health_index'])


# In[51]:


train_pct_index = int(0.80 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]


# In[52]:


#preparing random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[53]:


#training model
rf.fit(X_train, y_train);


# In[54]:


#making predictions on test set
predictions = rf.predict(X_test)
predictions = predictions.round()
#sort ettik
#predictions = np.sort(predictions)
#y_test = np.sort(y_test)


# In[55]:


#Calculating errors
errors = abs(predictions - y_test)


# In[56]:


#Mean absolute error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[57]:


#Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)


# In[58]:


# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[59]:


import matplotlib.pyplot as plt
plt.plot(range(len(predictions)),predictions,color  = "blue")
plt.plot(range(len(y_test)),y_test,color  = "red", alpha = 0.4)


# In[62]:


#Naive bayes
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[65]:


#SVM
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, y_test) 
accuracy


# In[ ]:




