import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor

#boston dataset extended: added features using PolynomialFaetures
def load_extended_boston():
    boston = datasets.load_boston()
    X1 = boston.data
    X1 = MinMaxScaler().fit_transform(boston.data)
    X1 = PolynomialFeatures(degree = 2, include_bias= False).fit_transform(X1)
    return X1, boston.target

#split the data into train and test, train the linear model and find R-squared
X1,y1 = load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X1,y1, random_state= 0)
lr = LinearRegression().fit(X_train, y_train)
print("Training set score: {:-2f}".format(lr.score(X_train, y_train)))  #0.95
print("Testing set score: {:-2f}".format(lr.score(X_test, y_test)))  #0.60

#The reason behind diffrent test and train scores might be due to overfitting or multicollinearity


