# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

dataset.head()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting the simple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set
y_predict= regressor.predict(X_test)

#visulising the training set results
plt.scatter(X_train, y_train, color='red')
plt.title('Train set')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

#visulising the test set results
plt.scatter(X_test, y_test, color='red')
plt.title('Test set prediction')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()
