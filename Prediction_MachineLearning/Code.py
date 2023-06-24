#!/usr/bin/env python
# coding: utf-8

# Importing all the required and necessary libraries.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# # Dataset

# In[2]:


# reading and exploring the dataset
dataset = pd.read_csv('Data_Set.csv')


# In[3]:


print(dataset)


# # Exploratory Data Analysis

# In[4]:


dataset.shape


# In[5]:


dataset.info()


# In[6]:


# viewing the description of the dataset to get a deep insight
dataset.describe()


# ### Correlation
# 

# In[7]:


# observing correlation relationship in the dataset
plt.figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(dataset.corr(), annot = True, cmap = 'terrain')


# In[8]:


cor = dataset.corr()
cor['TARGET_LifeExpectancy'].sort_values(ascending=False)


# ### Estimating distribution of Variable using Histogram plot

# In[9]:


# exploring distribution of variables using histogram
plt.figure(figsize=(30,30))
for i, col in enumerate(dataset.columns):
    plt.subplot(5,5,i+1, label=f"subplot{i}")
    plt.hist(dataset[col], alpha=0.3, color='b', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# ### Exploring each column in the dataset

# #### Target Life Expectancy Column

# In[10]:


#exploring the target Life expectancy column will give us the Maximum, Minumum, Median, Quartiles of the life expectancy of devices in the dataset
dataset.boxplot(column=['TARGET_LifeExpectancy'], return_type='axes')


# #### Company Column

# In[45]:


# exploring the company/ country column will give us the number of devices under each unique code - company code. 
plt.figure(figsize=(30,30))
ax = dataset['Country'].value_counts().plot(kind='bar')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.xlabel('Country', rotation=90)
plt.show()


# #### Year Column

# In[12]:


# exploring the year column will give us the number of devices under each year. 
dataset['Year'].value_counts().plot(kind = 'bar')


# #### Company Status

# In[13]:


# exploring the company status column will give us the data of devices under developing = 0, or developed = 1 status. 
dataset['Company_Status'].value_counts().plot(kind = 'bar')


# #### Company Confidence Column

# In[14]:


#exploring the company confidence column will give us the Maximum, Minumum, Median, Quartiles of the company confidence of devices in the dataset
dataset.boxplot(column=['Company_Confidence'], return_type='axes')


# #### Company Device Confidence Column

# In[15]:


#exploring the company device confidence column will give us the Maximum, Minumum, Median, Quartiles of the company device confidence of devices in the dataset
dataset.boxplot(column=['Company_device_confidence'], return_type='axes')


# #### Device Confidence Column

# In[16]:


#exploring the device confidence column will give us the Maximum, Minumum, Median, Quartiles of the device confidence of devices in the dataset
dataset.boxplot(column=['Device_confidence'], return_type='axes')


# #### Device Return Column

# In[17]:


#exploring the device return column will give us the Maximum, Minumum, Median, Quartiles of the device return of devices in the dataset
dataset.boxplot(column=['Device_returen'], return_type='axes')


# #### Test Fail Column

# In[18]:


#exploring the test fail column will give us the Maximum, Minumum, Median, Quartiles of the test fail of devices in the dataset
dataset.boxplot(column=['Test_Fail'], return_type='axes')


# #### Percentage Expenditure Column

# In[19]:


#exploring the percentage expenditure column will give us the Maximum, Minumum, Median, Quartiles of the percentage expenditure of devices in the dataset
dataset.boxplot(column=['PercentageExpenditure'], return_type='axes')


# #### Engine Cooling Column

# In[20]:


#exploring the engine cooling column will give us the Maximum, Minumum, Median, Quartiles of the engine cooling of devices in the dataset
dataset.boxplot(column=['Engine_Cooling'], return_type='axes')


# #### Gas Pressure

# In[21]:


#exploring the gas pressure column will give us the Maximum, Minumum, Median, Quartiles of the gas pressure of devices in the dataset
dataset.boxplot(column=['Gas_Pressure'], return_type='axes')


# #### Obsolescence Column

# In[22]:


#exploring the Obsolescence column will give us the Maximum, Minumum, Median, Quartiles of the Obsolescence of devices in the dataset
dataset.boxplot(column=['Obsolescence'], return_type='axes')


# #### Total Expenditure Column

# In[23]:


#exploring the total expenditure column will give us the Maximum, Minumum, Median, Quartiles of the total expenditure of devices in the dataset
dataset.boxplot(column=['TotalExpenditure'], return_type='axes')


# #### ISO 23 Column

# In[24]:


#exploring the ISO 23 column will give us the Maximum, Minumum, Median, Quartiles of the ISO 23 of devices in the dataset
dataset.boxplot(column=['ISO_23'], return_type='axes')


# #### STRD_DTP Column

# In[25]:


#exploring the STRD_DTP column will give us the Maximum, Minumum, Median, Quartiles of the STRD_DTP of devices in the dataset
dataset.boxplot(column=['STRD_DTP'], return_type='axes')


# #### Engine Failure Column

# In[26]:


#exploring the engine failure column will give us the Maximum, Minumum, Median, Quartiles of the engine failure of devices in the dataset
dataset.boxplot(column=['Engine_failure'], return_type='axes')


# #### GDP Column

# In[27]:


#exploring the GDP column will give us the Maximum, Minumum, Median, Quartiles of the GDP of devices in the dataset
dataset.boxplot(column=['GDP'], return_type='axes')


# #### Product Quantity Column

# In[28]:


#exploring the Product Quantity will give us the Maximum, Minumum, Median, Quartiles of the product quantity of devices in the dataset
dataset.boxplot(column=['Product_Quantity'], return_type='axes')


# #### Engine Failure Prevalance Column

# In[29]:


#exploring the engine failure prevalance column will give us the Maximum, Minumum, Median, Quartiles of the engine failure prevalance of devices in the dataset
dataset.boxplot(column=['Engine_failure_Prevalence'], return_type='axes')


# #### Leakage Prevalence Column

# In[30]:


#exploring the leakage prevalence column will give us the Maximum, Minumum, Median, Quartiles of the leakage prevalence of devices in the dataset
dataset.boxplot(column=['Leakage_Prevalence'], return_type='axes')


# #### Income Composition Of Resources Column

# In[31]:


#exploring the income composition of resources column will give us the Maximum, Minumum, Median, Quartiles of the income composition of resources of devices in the dataset
dataset.boxplot(column=['IncomeCompositionOfResources'], return_type='axes')


# #### RD Column

# In[32]:


#exploring the RD column will give us the Maximum, Minumum, Median, Quartiles of the RD of devices in the dataset
dataset.boxplot(column=['RD'], return_type='axes')


# ### Relationship between Target Variable and other Variables

# #### Scatter Plot

# In[33]:


# Plotting a scatter plot to explore the relationship between target and other variables. 
import seaborn as sns
plt.figure(figsize=(30,30))
for i, col in enumerate(dataset.columns):
    plt.subplot(5,5,i+1)
    sns.scatterplot(data=dataset, x=col, y='TARGET_LifeExpectancy')
    plt.title(col)
plt.xticks(rotation='vertical')
plt.show()


# #### Regplot 

# In[34]:


# Plotting a scatter plot to determine the linear relationship with the help of a line between target and other variables.
import seaborn as sns
plt.figure(figsize=(30,30))
for i, col in enumerate(dataset.columns):
    plt.subplot(5,5,i+1)
    sns.regplot(x=col,y='TARGET_LifeExpectancy', data=dataset)
    plt.title(col)
plt.xticks(rotation='vertical')
plt.show()


# # Refining the Dataset for Modelling

# In[35]:


# Dropping ID column, as this is not a feature. 
dataset = dataset.drop(columns='ID')
dataset.head()


# In[36]:


dataset_X = dataset.drop(['TARGET_LifeExpectancy'], axis=1)
dataset_y = dataset['TARGET_LifeExpectancy']


# In[37]:


# Splitting the dataset into train and test.
from sklearn.model_selection import train_test_split
dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=0)


# # Feature Scaling

# In[38]:


# defining attributes for min max scaling 
minmax_attributes = ['Test_Fail','IncomeCompositionOfResources','Gas_Pressure','Country', 'Year', 'Company_Status']

# defining attributes for power transform 
logNorm_attributes = ['Company_Confidence', 'Company_device_confidence', 'Device_confidence',
       'Device_returen','PercentageExpenditure','Engine_Cooling','Obsolescence', 'ISO_23',
       'TotalExpenditure', 'STRD_DTP', 'Engine_failure', 'GDP',
       'Product_Quantity', 'Engine_failure_Prevalence', 'Leakage_Prevalence','RD']

dataset_X_train_scaled = dataset_X_train.copy()
dataset_X_test_scaled = dataset_X_test.copy()

# performing min max scaling on the min max attributes
minmaxscaler = MinMaxScaler().fit(dataset_X_train_scaled.loc[:, minmax_attributes])
dataset_X_train_scaled.loc[:, minmax_attributes] = minmaxscaler.transform(dataset_X_train_scaled.loc[:, minmax_attributes])
dataset_X_test_scaled.loc[:, minmax_attributes] = minmaxscaler.transform(dataset_X_test_scaled.loc[:, minmax_attributes])

# performing power transform on the log norm attributes
powertransformer = PowerTransformer(method='yeo-johnson', standardize=False).fit(dataset_X_train.loc[:, logNorm_attributes])
dataset_X_train_scaled.loc[:, logNorm_attributes] = powertransformer.transform(dataset_X_train.loc[:, logNorm_attributes])
dataset_X_test_scaled.loc[:, logNorm_attributes] = powertransformer.transform(dataset_X_test.loc[:, logNorm_attributes])

# performing min max scaling on the log norm attributes
minmaxscaler_pt = MinMaxScaler().fit(dataset_X_train_scaled.loc[:, logNorm_attributes])
dataset_X_train_scaled.loc[:, logNorm_attributes] = minmaxscaler_pt.transform(dataset_X_train_scaled.loc[:, logNorm_attributes])
dataset_X_test_scaled.loc[:, logNorm_attributes] = minmaxscaler_pt.transform(dataset_X_test_scaled.loc[:, logNorm_attributes])


# We can use plots to see if everything is in order and are identically distributed.

# In[39]:


# Exploring if everything is in order after splitting and scaling
plt.figure(figsize=(20,20))
for i, col in enumerate(dataset_X_train_scaled.columns):
    plt.subplot(5,5,i+1)
    plt.hist(dataset_X_train_scaled[col], alpha=0.3, color='b', density=True)
    plt.hist(dataset_X_test_scaled[col], alpha=0.3, color='r', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# # Model 1 - After Feature Scaling

# ### Simple Linear Regression Model

# In[40]:


# Simple linear regression model
model = LinearRegression()

# Fitting the training data to the model
model.fit(dataset_X_train_scaled, dataset_y_train)

# Using test set to make the predictions
dataset_y_pred = model.predict(dataset_X_test_scaled)

# Calculate R-squared and the MSE of the model
mse = mean_squared_error(dataset_y_test, dataset_y_pred)
r2 = r2_score(dataset_y_test, dataset_y_pred)

# Using Mean square error and r2 score to evaluate the performance of the model
print("MSE: {:.3f}".format(mse))
print("R-squared: {:.3f}".format(r2))


# ### Polynomial Regression Model

# In[41]:


# Polynomial regression Model, setting degree to 2.
poly_model = PolynomialFeatures(degree=2)
dataset_X_poly_train = poly_model.fit_transform(dataset_X_train_scaled)
dataset_X_poly_test = poly_model.fit_transform(dataset_X_test_scaled)
model = LinearRegression()

# Fitting the training data to the model
model.fit(dataset_X_poly_train, dataset_y_train)

# Using test set to make the predictions
dataset_y_pred = model.predict(dataset_X_poly_test)

# Calculate R-squared and the MSE of the model
mse = mean_squared_error(dataset_y_test, dataset_y_pred)
r2 = r2_score(dataset_y_test, dataset_y_pred)

# Using Mean square error and r2 score to evaluate the performance of the model
print("MSE: {:.3f}".format(mse))
print("R-squared: {:.3f}".format(r2))


# ### Code Reference
# 
# 1. https://data36.com/polynomial-regression-python-scikit-learn/ Date Accessed: 25-03-2023
# 2. https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/ Date Accessed: 25-03-2023
# 3. https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386 Date Accessed 25-03-2023
# 4. https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression Date Accessed: 28-03-2023

# # Regularisation, Hyperparameter tuning and K Fold cross validation

# ## Model 2 - Lasso Regression 

# In[42]:


# Pipeline defining, and increasing the iterations to 10000
model = make_pipeline(PolynomialFeatures(), Lasso(max_iter=10000))

# for search over, we define the hyperparameters
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'lasso__alpha': np.logspace(-5, 1, num=25)
              }

# cross validation scheme definition
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid search 
grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(dataset_X_train_scaled, dataset_y_train)

# using test set to make predictions 
y_test_pred = grid_search.predict(dataset_X_test_scaled)

# calculate the mean squared error (MSE) on the test set
test_mse = mean_squared_error(dataset_y_test, y_test_pred)

# calculate the R2 score on the test set
r2 = r2_score(dataset_y_test, y_test_pred)

# Best hyperparameters to be printed out
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# Using Mean square error and r2 score to evaluate the performance of the model
print('MSE:', test_mse)
print('R-squared:', r2)


# ### Code Reference
# 
# 1. https://www.kaggle.com/code/deepakdodi/lasso-and-ridge-hypertuning-over-gapminder-dataset/notebook Date Accessed: 28-03-2023
# 2. https://medium.com/mlearning-ai/lasso-regression-and-hyperparameter-tuning-using-sklearn-885c78a37a70 Date Accessed: 28-03-2023
# 3. https://machinelearningmastery.com/lasso-regression-with-python/ Date Accessed: 28-03-2023
# 4. https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression Date Accessed: 28-03-2023

# ## Model 3 - Ridge Regression

# In[43]:


# Define the pipeline with Ridge regression
model = make_pipeline(PolynomialFeatures(), Ridge())

# for search over, we define the hyperparameters
param_grid = {'ridge__alpha': np.logspace(-5, 1, num=25),
              'polynomialfeatures__degree': [1, 2, 3]}

# cross validation scheme definition
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid search 
grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(dataset_X_train_scaled, dataset_y_train)

# using test set to make predictions 
y_pred = grid_search.predict(dataset_X_test_scaled)

# Calculate the MSE and R-squared of the model on the test set
mse = mean_squared_error(dataset_y_test, y_pred)
r2 = r2_score(dataset_y_test, y_pred)

# Best hyperparameters to be printed out
print('Best hyperparameters:', grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# Using Mean square error and r2 score to evaluate the performance of the model
print('MSE:', mse)
print('R-squared:', r2)


# ### Code Reference
# 
# 1. https://stackoverflow.com/questions/57376860/how-to-run-gridsearchcv-with-ridge-regression-in-sklearn Date Accessed: 31-03-2023
# 2. https://machinelearningmastery.com/ridge-regression-with-python/ Date Accessed: 31-03-2023
# 3. https://datascience.stackexchange.com/questions/66389/ridge-regression-model-creation-using-grid-search-and-cross-validation Date Accessed: 31-03-2023
# 4. https://www.kaggle.com/code/deepakdodi/lasso-and-ridge-hypertuning-over-gapminder-dataset/notebook Date Accessed: 31-03-2023
# 5. https://machinelearninghd.com/ridgecv-regression-python/ Date Accessed: 31-03-2023
# 6. https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression Date Accessed: 28-03-2023

# ## Saving Predictions File 

# In[44]:


# Reading and writing the predictions of Life expectancy into a csv file. 
dummy_x = dataset_y_test.reset_index()
predictions = pd.DataFrame()
predictions['ID'] = dummy_x['index']
predictions['Target'] = y_test_pred
predictions.to_csv('s3828461.csv')

