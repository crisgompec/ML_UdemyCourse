# Simple Linear Regression

"""
ASUMPTIONS OF A LINEAR REGRESSION:
    - Linearity
    - Homoscdasticity
    - Multivariate normality 
    - Independence of errors
    - Lack of multicollinearity

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Importamos el MODELO DE REGRESIÓN LINEAL desde sklearn
regressor = LinearRegression()
regressor.fit(X_train, y_train)

""" Aquí el modelo ya esta creado, ahora el objetivo es usar la parte del 
dataset orientada a testear si el modelo es adecuado o no. 'El regressor aprende
 de los datos que le proporcionas'. """

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()