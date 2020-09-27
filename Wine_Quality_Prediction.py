#!/usr/bin/env python
# coding: utf-8

# # Predicting Wine Color

# ## Project Description
# 
# We will be using the wine quality data set for this project and the dataset is available online at https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/. Our goal is to predict the color of the wine using different features. This data set contains various chemical properties of wine, such as acidity, sugar, pH, and alcohol. It also contains a quality metric (3-9, with highest being better) and a color (red or white). The name of the file is `Wine_Quality_Data.csv`.

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np

#Loading the dataset
filepath= 'C:/Users/pricr/Downloads/data/data/Wine_Quality_Data.csv'
data = pd.read_csv(filepath)


# In[2]:


data.head()


# In[3]:


data.dtypes


# Convert the color feature to an integer. This is a quick way to do it using Pandas.

# In[4]:


data['color'] = data.color.replace('white',0).replace('red',1).astype(np.int)


# ## Creating Training and Test Sets

# In[5]:


# All data columns except for color
feature_cols = [x for x in data.columns if x not in 'color']


# In[6]:


from sklearn.model_selection import StratifiedShuffleSplit

# Split the data into two parts with 1000 points in the test data
# This creates a generator
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)

# Get the index values from the generator
train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols], data['color']))

# Create the data sets
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'color']

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'color']


# In[7]:


y_train.value_counts(normalize=True).sort_index()


# In[8]:


y_test.value_counts(normalize=True).sort_index()


# In[9]:


X_train.head()


# In[10]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)


# The number of nodes and the maximum actual depth.

# In[11]:


dt.tree_.node_count, dt.tree_.max_depth


# A function to return error metrics.

# In[12]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},
                      name=label)


# The decision tree predicts a little better on the training data than the test data, which is consistent with (mild)  overfitting. Also notice the perfect recall score for the training data. In many instances, this prediction difference is even greater than that seen here. 

# In[13]:


# The error on the training and test data sets
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                              measure_error(y_test, y_test_pred, 'test')],
                              axis=1)

train_test_full_error


# In[14]:


from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),
              'max_features': range(1, len(dt.feature_importances_)+1)}

GR = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)

GR = GR.fit(X_train, y_train)


# The number of nodes and the maximum depth of the tree.

# In[15]:


GR.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth


# These test errors are a little better than the previous ones. So it would seem the previous example overfit the data, but only slightly so.

# In[16]:


y_train_pred_gr = GR.predict(X_train)
y_test_pred_gr = GR.predict(X_test)

train_test_gr_error = pd.concat([measure_error(y_train, y_train_pred_gr, 'train'),
                                 measure_error(y_test, y_test_pred_gr, 'test')],
                                axis=1)


# In[17]:


train_test_gr_error


# In[18]:


feature_cols = [x for x in data.columns if x != 'residual_sugar']

# Create the data sets
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'residual_sugar']

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'residual_sugar']


# In[19]:


from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor().fit(X_train, y_train)

param_grid = {'max_depth':range(1, dr.tree_.max_depth+1, 2),
              'max_features': range(1, len(dr.feature_importances_)+1)}

GR_sugar = GridSearchCV(DecisionTreeRegressor(random_state=42),
                     param_grid=param_grid,
                     scoring='neg_mean_squared_error',
                      n_jobs=-1)

GR_sugar = GR_sugar.fit(X_train, y_train)


# The number of nodes and the maximum depth of the tree. This tree has lots of nodes, which is not so surprising given the continuous data.

# In[20]:


GR_sugar.best_estimator_.tree_.node_count, GR_sugar.best_estimator_.tree_.max_depth


# The error on train and test data sets. Since this is continuous, we will use mean squared error.

# In[21]:


from sklearn.metrics import mean_squared_error

y_train_pred_gr_sugar = GR_sugar.predict(X_train)
y_test_pred_gr_sugar  = GR_sugar.predict(X_test)

train_test_gr_sugar_error = pd.Series({'train': mean_squared_error(y_train, y_train_pred_gr_sugar),
                                         'test':  mean_squared_error(y_test, y_test_pred_gr_sugar)},
                                          name='MSE').to_frame().T

train_test_gr_sugar_error


# A plot of actual vs predicted residual sugar.

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook')
sns.set_style('white')
sns.set_palette('dark')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


fig = plt.figure(figsize=(6,6))
ax = plt.axes()

ph_test_predict = pd.DataFrame({'test':y_test.values,
                                'predict': y_test_pred_gr_sugar}).set_index('test').sort_index()

ph_test_predict.plot(marker='o', ls='', ax=ax)
ax.set(xlabel='Test', ylabel='Predict', xlim=(0,35), ylim=(0,35));


# In[ ]:




