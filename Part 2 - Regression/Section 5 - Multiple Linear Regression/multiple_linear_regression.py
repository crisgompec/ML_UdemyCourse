# Multiple Linear Regression

"""
DUMMY VARIABLES LECTURE: 
Al pasar de datos categoricos a dummy variables, se verifica
que dichas dummy variables estan relacionadas de las siguiente manera:
    D1=1-D2-D3...
Por tanto, para preservar la Lack of multicollinearity (caracteristica necesaria) para
un buen modelo de regresión, es necesario usar siempre una variable dummy menos del
total que hay disponibles.

BACKWARD ELIMINATION:
1) Select significance level (SL) limit for the variables to stay in model (que expliquen el modelo)
2) Fit the full model with all possible predictors (independent variables)
3) Consider the predictor with highest P-value and see if P>SL. If so, remove variable. O.W., you're ready
4) Fit the model without variable. Go to step 3

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')# indicamos la columna de la categoria
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # Eliminates one dummy variable (california). Queda NY y Fl.

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, t est_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results: TO SEE PERFOMANCE! (Metrics will be taught later on)
y_pred = regressor.predict(X_test)

# APPLYING BACKWARD ELIMINATION TO GET A BETTER MODEL
import statsmodels.api as sm # Para movidas de p-value
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

# X_opt obtains the ind variables that truly impact the final outcome
X_opt = X[:, [0,1,2,3,4,5]] # We indicate the independent variables to be tested for being removed from model
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Orderly least squares?
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]] # We indicate the independent variables to be tested for being removed from model
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Orderly least squares?
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]] # We indicate the independent variables to be tested for being removed from model
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Orderly least squares?
regressor_OLS.summary()

X_opt = X[:, [0,3,5]] # We indicate the independent variables to be tested for being removed from model
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Orderly least squares?
regressor_OLS.summary()

X_opt = X[:, [0,3]] # We indicate the independent variables to be tested for being removed from model
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Orderly least squares?
regressor_OLS.summary()
