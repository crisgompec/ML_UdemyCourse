# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd # La mejor libreria para importar dataset y manage them

# IMPORTING DATA
dataset = pd.read_csv('Data.csv')

# Select independent variable vectors
X = dataset.iloc[:, :-1].values # iloc: para seleccionar filas y columnas de datos a incorporar
# Select Dependent variable vector
y = dataset.iloc[:, 3].values

# TAKING CARE OF MISSING DATA - NUMERICAL
from sklearn.impute import SimpleImputer # This is a class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', )

# Sustituimos los 'nan' por la media de valores para salary y age.
imputer.fit(X[:,1:3]) # En python, los indexes de uperbound son hasta el anterior (2, no 3)
X[:,1:3] = imputer.transform(X[:,1:3])

# CATEGORICAL VARIABLES (STRINGS): HOW TO ENCODE THEM INTO NUMBERS
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0]) # Solo pillamos 'Country'
X[:,0] = labelencoder_X.fit_transform(X[:,0]) # Nos da Fracia = 0, por ejemplo

# Para evitar que los algoritmos diga España>Francia, usamos 0 y 1s (dummy variable)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')# indicamos la columna 
X = ct.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# PREPARE TRAIN-TEST DATASETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling --> Cuanodo el orden de magnitud de las variables independientes cambia
# mucho de una variable a otra (puede influenciar en los algoritmos de ML). Se resta media y 
# y se divide por la desviacion típica.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)""" # LA Y NO HACE FALTA POR SER CATEGORICA!









