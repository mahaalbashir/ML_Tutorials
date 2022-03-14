import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm

#boston dataset extended: added features using PolynomialFaetures
def load_extended_boston():
    boston = datasets.load_boston()
    X = boston.data
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree = 2, include_bias= False).fit_transform(X)
    return X, boston.target

#split the data into train and test, train the linear model with and without Ridge
X,y = load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state= 0)
lr = LinearRegression().fit(X_train, y_train)
#print("Training set score: {:-2f}".format(lr.score(X_train, y_train)))  #0.95
#print("Testing set score: {:-2f}".format(lr.score(X_test, y_test)))  #0.60

#calcutating adjusted R-squared
def adjusted_R_squared(X,y,lr):
    adj_r2 = 1 - (1-lr.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)
    return adj_r2

print("Training set adjusted R-squared: {:-2f}".format(adjusted_R_squared(X_train,y_train,lr)))  #0.93
print("Testing set adjusted R-squared: {:-2f}".format(adjusted_R_squared(X_test,y_test,lr)))  #-1.25
print(len(y_train))


#plot matrix for the set of features/attributes 
X_plot = pd.DataFrame(X)
#pd.plotting.scatter_matrix(X_plot) 

#produce the VIF table for the independant variables
# VIF dataframe
vif = pd.DataFrame()

# calculating VIF for each feature
vif["VIF"] = [variance_inflation_factor(X_plot.values, i) for i in range(len(X_plot.columns))]
#print(vif)

#using ridge
ridge = Ridge(alpha = 1.0).fit(X_train, y_train)
#print("Ridge Training set score: {:-2f}".format(ridge.score(X_train, y_train)))  #0.89
#print("Ridge Testing set score: {:-2f}".format(ridge.score(X_test, y_test)))  #0.75

#tune alpha hyper parameter in ridge
for i in range (0,11):
    ridge = Ridge(alpha = i).fit(X_train, y_train)
    #print("Ridge Training set score when alpha is", i, ": {:-2f}".format(ridge.score(X_train, y_train)))
    #print("Ridge Testing set score when alpha is", i, ": {:-2f}".format(ridge.score(X_test, y_test))) 
    visualizer = ResidualsPlot(ridge)
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    #visualizer.show()                 # Finalize and render the figure

#The model best performance when alpha is one then the testing score starts to decrease

#using Lasso
lasso = Lasso(alpha = 1.0).fit(X_train, y_train)
#print("Lasso Training set score: {:-2f}".format(lasso.score(X_train, y_train)))  #0.29
#print("Lasso Testing set score: {:-2f}".format(lasso.score(X_test, y_test)))  #0.21

#tune alpha hyper parameter in Lasso
for i in range (0,11):
    lasso = Lasso(alpha = i).fit(X_train, y_train)
    #print("Lasso Training set score when alpha is", i, ": {:-2f}".format(lasso.score(X_train, y_train)))
    #print("Lasso Testing set score when alpha is", i, ": {:-2f}".format(lasso.score(X_test, y_test))) 

#The model performs badly in general. The best performance is when alpha = 1 with 0.29 for training and 0.21 for testing
#the model score is 0 and -0.001 for trianing and testing whne alpha > 1

