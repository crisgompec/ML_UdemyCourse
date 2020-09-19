# Artificial Neural Network
"""
Mirate el esquemita de inputs, neurons y outputs.

Activation function: toman como input la suma pesada de las inputs
y producen una output acotada entre -1/0 y 1. Algunas de las activation 
functions:
    - Threshold
    - Sigmoid
    - Rectifier
    - Hyperbolic tangent
Como elegirlas? Depende. Si las inputs son bool, puedes elegir
threshold o sigmoid porque estan acotadas entre 0 y 1.

How does a neuron work -> amazing video

How does a neuron netwoks learn? Try values, evalue a cost function,
update weights to improve it. Multiple iterations until the cost 
function is minimized. Este metodo se llama backpropagations.

Gradient descent: forma inteligente de encontrar el minimo de la 
funcion de coste C = f(y_hat) evaluando la pendiente de la funcion
y avanzar hacia el lado que apunta abajo. 

El problema de esto es que solo funciona con problemas convexos. 
(seriamos capaces de detectar el minimo local pero no local). La
solucion: stocastic gradiente descent. Con este metodo, si ajustan
los pesos con cada dato [una fila] (no a la vez todas las filas de datos).
Faster: only needs to load one row data. Pero es más random, cada ronda
sera distinta y actualizaras la red de forma distinta.



"""
# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Splitting the dataset   into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# Rectifier activation function is proved to be one the best one for hidden layers.
# Optimum number of nodes: tip: avg. of number of nodes input and output layer
# tuning: tells you what is the optimal parameters of model
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # input_dim=12

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
# Optimizer: algorithm to find the best weights to make ann more powerful -> we choose stochastic gradient descent 'adam'
# loss: loss function (can be the sum of squared errors) or the logaritmic loss (una to rara que se asocia a sigmoid resutls)
# Metrics: in brackets because a list is needed
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
# the batch size is the number of observations before updating weights
# epoc: number of rounds to perform the back propagation (7 steps)
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)