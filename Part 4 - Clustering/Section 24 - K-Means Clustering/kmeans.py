# K-Means Clustering
"""
Clustering se parece a clasification, con la diferencia que NO sabemos lo que 
estamos intentando clasificar. Es muy interesante para descubrir grupos de 
datos que jamas hubieses considerado antes.

Antes, en clasificacion, sabiamos los valores de X e y, y aprendiendo sobre que 
valores de X nos daban y, elaborabamos un modelo para predecir y a partir de una 
nueva X.

En clustering, tenemos los datos de las X, PERO no sabemos nada de y. Intentamos 
ver si hay correlación entre las variables de entrada X que puedan producir una 
determinada y que no sabemos. ES DECIR, IDENTICAR AQUELLOS VALORES DE X QUE ESTAN
RELACIONADOS DE ALGUNA FORMA. 

K-Means es un algoritmo para conseguir hallar estos cluster, o grupos de datos.

Algortimo:
    1) Selecciono en cuantos K clusters quiero dividir mis datos
    2) Seleciono K centroides (punto no necesariamente de los datos) de forma aleatoria
    3) Asigno cada dato al centroide más proximo (formando K clusters)
    4) Computo y coloco nuevo centroide para cada cluster (como calcular
    el centro de gravedad del conjunto anterior de datos).
    5) Reasigno datos a cada nuevo centroide. Si no hubo cambios, ya esta hecho.
       Si hubo cambios, vuelvo a paso 4
       
Importante: la inicializacion tiene un efecto importante en los clusters finales 
que tenemos. ¿Como combatirlo? Hay una modificacion del algoritmo que se llama
k-means++ que basicamente te permite seleccionar de forma adecuada la inicializacion
de los centroides.    

Choosing the right number of clusters? We calculate the WCSS: sum_j(sums_i(Distance(P_j,C_i))).
Pj es punto en los datos. Ci es el centroide. We minimize it for all the number of cluster
we think would be suitable.

La mejor es si WCSS=0, cuando K=n_datos. Cuantos más clusters menor WCSS. Aunque 
mejore con cada Ci, cuanto esta mejorando realmente el WCSS? Bueno, sigue una 
exponencial negativa, normalmente los saltos de 3 a 4 ya no disminuye tanto. 
Donde más mejora es al pasar de 1 a 2 o de 3 a 4. -> 'Elbow method': Look where 
is the elbow, before the improvemente is not that great after adding a cluster.

Problem: segment customers based on annual income and spending score.
Since we dont know how many cluster might be, it is a problem of clustering.

"""
#%reset -f
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # la inercia es lo que hemos definido como sum(distancias ci,pj)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X) # OJO, FIT_PREDICT, NO FIT ASECAS. 
# y_means es vector asociado a cada dato de X, dando un numero que se asocia al 
# cluster al que pertenece!

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()