#code source: Jaques Grobler
#Licence: BSD 3 clause
#changes made for teaching purpose by Hisham Ihshaish

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

#load the diabetes dataset
diabetes = datasets.load_diabetes()
#print(diabetes.keys())

data = pd.DataFrame(diabetes.data, columns = [diabetes.feature_names])
target = pd.DataFrame(diabetes.target)
#print(diabetes.data)
#print(target)

#plot the bmi and diabetes progression
#plt.scatter(data['bmi'],target)
#plt.xlabel("bmi")
#plt.ylabel("diabetes")
#plt.show()

#choosing the bmi as X and shaping it to a vector using np.newaxis
X = diabetes.data[:, np.newaxis, 2]
y = np.array(target)

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state= 42)
lr = LinearRegression().fit(X_train,y_train)

print("lr.coeff_: {}".format(lr.coef_))  #975.27
print("lr.intercept_: {}".format(lr.intercept_))  #152.07

#make prediction using the test data and find R-squared
y_pred = lr.predict(X_test)
print("Coefficient of Determination R squared: %.2f" % r2_score(y_test, y_pred))  #0.32

#plot outputs
plt.scatter(X_test, y_test, color = "black")
plt.plot(X_test, y_pred, color = "blue", linewidth = 3)
plt.xlabel("X_test")
plt.ylabel("y_test")
plt.xticks(())
plt.yticks(())
plt.show()

#This model doesnt sem to be a good predictive model because it only explains 32% of the variations of y
#R-squared 0.32
