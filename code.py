#!/usr/bin/env python
# coding: utf-8

# # Campus Recruitment EDA and Regression Prediction

# ### Data Description 

# sl_no: Serial Number,
# gender: Gender- Male='M',Female='F',
# ssc_p: Secondary Education percentage- 10th Grade,
# ssc_b: Board of Education- Central/ Others,
# hsc_p: Higher Secondary Education percentage- 12th Grade,
# hsc_b: Board of Education- Central/ Others,
# hsc_s: Specialization in Higher Secondary Education,
# degree_p: Degree Percentage,
# degree_t: Under Graduation(Degree type)- Field of degree education,
# workex: Work Experience,
# etest_p: Employability test percentage ( conducted by college),
# specialisation: Post Graduation(MBA)- Specialization,
# mba_p: MBA percentage,
# status: Status of placement- Placed/Not placed,
# salary: Salary offered by corporate to candidates.

# ### The goal is to predict the salary of employees using regression models 


# ## Importing the Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import the dataset 

df = pd.read_csv("Placement_Data_Full_Class_1.csv")
df.head()

df.shape

df.columns

#there are 8 object data that needs to be converted
df.info()


# # Descriptive Data Analysis 

df.describe()

df.describe(include ='object')

df['status'].value_counts()

df = df.drop('sl_no',axis=1)
df['salary'] = df['salary'].fillna(0)
df.head()


# # Data Visualization
# plotting barplot
plt.style.use('ggplot')
plt.figure(figsize=(20,25))
plt.subplot(4,2,1)
df['gender'].value_counts().plot(kind='bar',title='gender')
plt.subplot(4,2,2)
df['ssc_b'].value_counts().plot(kind='bar',title='ssc_b')
plt.subplot(4,2,3)
df['hsc_b'].value_counts().plot(kind='bar',title='hsc_b')
plt.subplot(4,2,4)
df['hsc_s'].value_counts().plot(kind='bar',title='hsc_s')
plt.subplot(4,2,5)
df['degree_t'].value_counts().plot(kind='bar',title='degree_t')
plt.subplot(4,2,6)
df['workex'].value_counts().plot(kind='bar',title='workex')
plt.subplot(4,2,7)
df['specialisation'].value_counts().plot(kind='bar',title='specialisation')
plt.subplot(4,2,8)
df['status'].value_counts().plot(kind='bar',title='status')
plt.show()

# plotting scatter-plot
plt.figure(figsize=(20,20))
plt.subplot(5,2,1)
sns.scatterplot(data=df, x="ssc_p", y="salary", hue="status")
plt.subplot(5,2,2)
sns.scatterplot(data=df, x="hsc_p", y="salary", hue="status")
plt.subplot(5,2,3)
sns.scatterplot(data=df, x="degree_p", y="salary", hue="status")
plt.subplot(5,2,4)
sns.scatterplot(data=df, x="etest_p", y="salary", hue="status")
plt.subplot(5,2,5)
sns.scatterplot(data=df, x="mba_p", y="salary", hue="status")
plt.show()


# # Data Preprocessing

# ## Encoding of categorical values using Label Encoder

# using label encoder beacuse the below columns are ordinal attributes
# replaces the attributes with 0,1 and 2 in alphabetically appearing columns 
from sklearn.preprocessing import LabelEncoder
cols = ['gender', 'ssc_b', 'hsc_b','hsc_s', 'degree_t','workex','specialisation', 'status']
le = LabelEncoder()
df[cols] = df[cols].apply(le.fit_transform)
df.head()


# ## Extract the independent (input) and dependent (output) variable 

# extracting independent variable
X = df.iloc[:,:-1].values
# extracting dependent variable
Y = df.iloc[:,-1].values
print(X.shape)
print(Y.shape)


# ## Splitting the dataset into the Training and Testing sets 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, random_state = 100)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# # Normalization of Dataset

# ## Standardize the data

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train = (X_train-X_mean)/X_std

X_test = (X_test-X_mean)/X_std

print(X_train.shape, X_test.shape)


# # Training Model using Regression algorithms
# 
# ### 1. Linear Regressor
# ### 2. Decision Tree Regressor
# ### 3. Random Forest Regressor
# ### 4. XGBoost Regressor

# ## Importing Libraries 

import sklearn
import numpy as np
import pandas as pd
from math import sqrt
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# ## Linear Regression

l_reg = LinearRegression()
l_reg.fit(X_train,Y_train)

print("Train accuracy:", (l_reg.score(X_train,Y_train))*100)
print("Test accuracy:", (l_reg.score(X_test,Y_test))*100)

print(l_reg.coef_)
print("B0 =",l_reg.intercept_)

Y_pred = l_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)

r2_l = r2_score(Y_test, Y_pred)*100
rms_l = sqrt(mean_squared_error(Y_test, Y_pred))
mae_l = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_l} %")
print(f"Root mean squared error is {rms_l}")
print(f"Mean absolute error is {mae_l}")


# ## Visualizing the results 

plt.style.use('fivethirtyeight') 
plt.figure(figsize=(10,6))
plt.scatter(l_reg.predict(X_train), l_reg.predict(X_train)-Y_train, color = "green", s = 10, label = 'LR Train data') 
plt.scatter(l_reg.predict(X_test), l_reg.predict(X_test)-Y_test, color = "blue", s = 10, label = 'LR Test data') 
plt.hlines(y = 0, xmin = 0, xmax = 950000, linewidth = 2) 
plt.legend(loc = 'upper right') 
plt.title("Residual errors") 
plt.xlabel("Salary")
plt.ylabel("Error")
plt.show() 


# ## Decision Tree Regressor

d_reg = DecisionTreeRegressor()
d_reg.fit(X_train,Y_train)

print("Train accuracy:", (d_reg.score(X_train,Y_train))*100)
print("Test accuracy:", (d_reg.score(X_test,Y_test))*100)

Y_pred = d_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)

r2_d = r2_score(Y_test, Y_pred)*100
rms_d = sqrt(mean_squared_error(Y_test, Y_pred))
mae_d = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_d} %")
print(f"Root mean squared error is {rms_d}")
print(f"Mean absolute error is {mae_d}")


# ## Random Forest Regressor 

r_reg = RandomForestRegressor()
r_reg.fit(X_train,Y_train)

print("Training accuracy:",(r_reg.score(X_train,Y_train))*100)
print("Test accuracy:",(r_reg.score(X_test,Y_test))*100)

Y_pred = r_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)

r2_r = r2_score(Y_test, Y_pred)*100
rms_r = sqrt(mean_squared_error(Y_test, Y_pred))
mae_r = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_r} %")
print(f"Root mean squared error is {rms_r}")
print(f"Mean absolute error is {mae_r}")


# ## XGBoost Regressor 

x_reg = XGBRegressor()
x_reg.fit(X_train,Y_train)

print("Training accuracy:",(x_reg.score(X_train,Y_train))*100)
print("Test accuracy:",(x_reg.score(X_test,Y_test))*100)

Y_pred = x_reg.predict(X_test)
print(Y_test.shape, Y_pred.shape)

r2_x = r2_score(Y_test, Y_pred)*100
rms_x = sqrt(mean_squared_error(Y_test, Y_pred))
mae_x = mean_absolute_error(Y_test, Y_pred)
print(f"R^2 score of model is {r2_x} %")
print(f"Root mean squared error is {rms_x}")
print(f"Mean absolute error is {mae_x}")


# # Evaluation Table

models = pd.DataFrame({
    'Algorithm': ['Linear Regression','Decision Tree Regressor', 'Random Forest Regressor',  'XGBoost Regressor'],
    'R^2 Score': [ r2_l, r2_d, r2_r, r2_x],
    'RMS Score' : [rms_l, rms_d, rms_r, rms_x],
    'MAE Score' : [mae_l, mae_d, mae_r, mae_x]
})

models.sort_values(by = ['R^2 Score', 'RMS Score', 'MAE Score'], ascending = True)


# # Plotting the Residual Chart

plt.style.use('fivethirtyeight') 
plt.figure(figsize=(12,8))
plt.scatter(l_reg.predict(X_train), l_reg.predict(X_train)-Y_train, color = "green", s = 20, label = 'LR Train data') 
plt.scatter(l_reg.predict(X_test), l_reg.predict(X_test)-Y_test, color = "blue", s = 20, label = 'LR Test data') 
plt.scatter(d_reg.predict(X_train), d_reg.predict(X_train)-Y_train, color = "red", s = 20, label = 'DT R Train data') 
plt.scatter(d_reg.predict(X_test), d_reg.predict(X_test)-Y_test, color = "black", s = 20, label = 'DT R Test data') 
plt.scatter(r_reg.predict(X_train), r_reg.predict(X_train)-Y_train, color = "yellow", s = 20, label = 'RF R Train data') 
plt.scatter(r_reg.predict(X_test), r_reg.predict(X_test)-Y_test, color = "pink", s = 20, label = 'RF R Test data') 
plt.scatter(x_reg.predict(X_train), x_reg.predict(X_train)-Y_train, color = "orange", s = 20, label = 'XGB R Train data') 
plt.scatter(x_reg.predict(X_test), x_reg.predict(X_test)-Y_test, color = "white", s = 20, label = 'XGB R Test data') 
plt.hlines(y = 0, xmin = 0, xmax = 950000, linewidth = 2) 
plt.legend(loc = 'upper right') 
plt.title("Residual errors") 
plt.xlabel("Salary")
plt.ylabel("Error")
plt.show() 

# plotting barplot among the models for comparison
plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='R^2 Score',data=models)
plt.show()


# # Cross-Validation of Models using K-fold CV

# ## Cross-Validation of Linear Regression Model

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
l_reg = LinearRegression()
l_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(l_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = l_reg.predict(X_test)
r2_l = sklearn.metrics.r2_score(Y_test, Y_pred)
print(f'R^2 Score: {r2_l}')


# ## Cross-Validation of Decision Tree Regressor Model

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
d_reg = DecisionTreeRegressor()
d_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(d_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = d_reg.predict(X_test)
r2_d = sklearn.metrics.r2_score(Y_test, Y_pred)
print(f'R^2 Score: {r2_d}')


# ## Cross-Validation of Random Forest Model 

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
r_reg = RandomForestRegressor()
r_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(r_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = r_reg.predict(X_test)
r2_r = sklearn.metrics.r2_score(Y_test, Y_pred)
print(f'R^2 Score: {r2_r}')


# ## Cross-Validation of XGBoost Regressor Model 

# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=100, shuffle=True)
# create model
x_reg = XGBRegressor()
x_reg.fit(X_train, Y_train)
# evaluate model
scores = cross_val_score(x_reg, X_train, Y_train, scoring='r2', cv=cv)
print(f'Score Array list: {scores}') 
print('\n')
# report performance
Y_pred = x_reg.predict(X_test)
r2_x = sklearn.metrics.r2_score(Y_test, Y_pred)
print(f'R^2 Score: {r2_x}')


# # Analysing the cross-validation of models 

models = pd.DataFrame({
    'Algorithm': ['Linear Regression','Decision Tree Regressor', 'Random Forest Regressor',  'XGBoost Regressor'],
    'R^2 Score': [ r2_l, r2_d, r2_r, r2_x],
    })

models.sort_values(by = ['R^2 Score'], ascending = True)


# # Ploting the graph

plt.figure(figsize=(12,8))
sns.barplot(x='Algorithm',y='R^2 Score',data=models)
plt.show()
