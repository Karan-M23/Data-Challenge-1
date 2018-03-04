
# coding: utf-8

# In[1]:


# Importing pandas and numpy packages to help with data handling
import pandas as pd
import numpy as np


# In[2]:


# Importing training data as (data) and test data as (test)
data = pd.read_csv('/Users/Karan_M/Documents/STAT 441/Data Challenge 1/individual_data_train.csv')
test = pd.read_csv('/Users/Karan_M/Documents/STAT 441/Data Challenge 1/individual_data_test.csv')


# In[3]:


# For both training and test data we want to obtain dummy variables which help us deal with categorical data
data1 = pd.get_dummies(data, prefix = ['Workclass', 'Education', 'Marital Status','Occupation',
                                       'Relationship','Race','Sex','Native Country'], 
                       columns = ['Workclass', 'Education', 'Marital Status',
                                  'Occupation','Relationship','Race','Sex','Native Country'])
test1 = pd.get_dummies(test, prefix = ['Workclass', 'Education', 'Marital Status','Occupation',
                                       'Relationship','Race','Sex','Native Country'], 
                       columns = ['Workclass', 'Education', 'Marital Status','Occupation',
                                  'Relationship','Race','Sex','Native Country'])


# In[4]:


# One value that does not show up in the Native Country column 
# in the test data is a person from the Netherlands. 
# As a result in the new test1 data with dummy variables has one less column, 
# we fix this by adding a column of zeros to test 1
test1['Native Country_ Holand-Netherlands'] = 0


# In[5]:


# In the training data we want to seperate 'class' the target variable and the other variables 
parvars = data1.columns.values.tolist()
testvars = data1.columns.values.tolist()
class_col = data1['Class']
y = np.array(class_col)
rest = [i for i in parvars if i not in 'Class']
X_all = data1[rest]


# In[6]:


# We utilize L1 norm penalties for feature selection so we can get rid of irrelevant variables 
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_all, y)
model = SelectFromModel(lsvc, prefit=True)
X_final = model.transform(X_all)


# In[7]:


# We utilize the same feature selection method on the test data
Xtestdata = model.transform(test1)


# In[8]:


# We use Gradient Boosting as our classifier on the training data
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(random_state=0)
classifier.fit(X_final,y)


# In[9]:


# Predict the 'class' variable in the test data
y_pred = classifier.predict(Xtestdata)


# In[10]:


# Print results to a csv 
submission = pd.DataFrame(y_pred, columns=['class'])
submission.head()
submission.to_csv('/Users/Karan_M/Documents/STAT 441/Data Challenge 1/sub.csv')

