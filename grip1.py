#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# In[2]:


grip1=pd.read_csv(r'C:\Users\ANSUMAN\Downloads\GRIP-1.csv')
grip1


# In[3]:


grip1.dtypes


# In[26]:


sns.set_style('darkgrid')
sns.scatterplot(y= grip1['Scores'], x= grip1['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[28]:


sns.regplot(x= grip1['Hours'], y= grip1['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(grip1.corr())


# In[12]:


X=grip1[['Hours']]
y=grip1["Scores"]


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3,random_state=100)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lm=LinearRegression()


# In[18]:


lm.fit(X_train,y_train)


# In[29]:


hours=[9.25]
answer=lm.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# # According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 92.809 marks

# In[ ]:




