# Hierarchical Clustering
"""
Different type of clustering. 2 types: Agglomerative & Divisive. 

Let's focus on the Agglomerative HC.

Algorithm:
    1) Make every data point a cluster
    2) Merge the 2 closest clusters into one cluster -> N-1 cluster
    3) ... repeat 2 until we have K clusters
    4) Or until we have one 1 cluster left -> Creates Dendogram! [Como el inverso 
    de un decision tree].
    
Closest clusters? How to measure distance: EUCLIDEAN [From Centroids, or closest points,
or furthest points, or from avg. points]. 

Dendograms: X vs. Euclidean distances. How can we use them? Para elegir cuantos
clusters queremos. Nos permite ver con claridad cuanta va ser la distancia entre
que clusters (la distancia es el parametro que elegimos para obtener clusters, ya
no es el numero de clusters que queremos sino la distancia minima que queremos 
entre ellos).

We also can evaluate the number of clusters that we need by crossing the longest 
distance between a cluster merge. -> Criteria que se usa.

In this problem, we DONT know how many clusters we'll have. We'll use the dendogram
approach.

"""
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

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) # 'ward' minimizes variance in the clusters
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()