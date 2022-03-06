#!/usr/bin/env python
# coding: utf-8

# ## The Spark Foundation - Data Science & Business Analytics Internship
# ### Author - Vishal Bapuso Khandagale
# ### Batch - March 2022
# ### Task 1 - Prediction using supervised machine learning.
# ### Simple Linear Regression
# #### In this regression task we predict the percentage of marks that student is expwcted to score based upon thenumber of hours they studied. In this regression task it involves two variables.
# #### To predict : What will be the predicted score if a student studies for 9.25 hrs/day.
# #### Importing all libraries required in this jupyter notebook.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


dataset=pd.read_csv('http://bit.ly/w-data')


# In[6]:


print(dataset.shape)
2
dataset.head()


# #### Using datarset.describe to get the statistical summury of the dataframe

# In[8]:


dataset.describe()


# #### Using dataset.info to get a concise summary of the dataframe.

# In[10]:


dataset.info()


# #### plotting the distribution of score.

# In[12]:


dataset.plot(x='Hours',y='Scores',style='o')

plt.title('Hours vs Percentage')

plt.xlabel('Hours Studied')

plt.ylabel('Percentage Score')

plt.show()


# In[14]:


x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,1].values


# In[16]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()

regressor.fit(x_train,y_train)

print("Training Complete")


# #### Plotting the regression line

# In[20]:


line = regressor.coef_*x+regressor.intercept_


# #### Plotting the regression data 

# In[22]:


plt.scatter(x,y)

plt.plot(x,line);

plt.show()


# In[23]:


#Testing data in hours

print(x_test)

#Predicting the scores

y_pred=regressor.predict(x_test)


# #### compairing Actual vs Predicted 

# In[24]:


df = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
df


# #### visualising the training set result 

# In[25]:


plt.scatter(x_train,y_train,color='green')

plt.plot(x_train, regressor.predict(x_train),color='red')

plt.title('Studied Hours vs Persentage scores \n Graph')

plt.xlabel('Hours of Studied')

plt.ylabel('Percentage of Marks')

plt.show()


# #### visualising the test results 

# In[29]:


dataset=np.array(9.25)
dataset=dataset.reshape(-1, 1)
pred=regressor.predict(dataset)

print("If student studies for 9.25 hours/day ,the score is",format(pred))


# In[31]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[33]:


from sklearn.metrics import r2_score
print("The R-Square of the  model is: ", r2_score(y_test, y_pred))


# ### Conclusion -
# ### What will be predicted score if a student studies for 9.25 hours/day 
# #### if a student studies for 9.25 hours/day
# #### the predicted Score came out to  be 93.69
