# Apriori
"""
"Si persona X ha visto una pelicula Y, seguramente ha visto tambien la pelicula Z"

IMP CONCEPTS
Support:    people that watched movie Y / total people
Confidence: people that watched movie Y / total people that watched movie Z
Lift:       Confidence/Support (como en Naive-Bayes)

Así, identificamos poblaciones cuya probabilidad es muy alta en comparacion al
analisis general de la poblacion. (valores altos de lift). It's the improvement
with respect to the previous model.

Algorithm:
    1) Set support and confidence
    2) Take subsets having higher support than minimum support
    3) Take all the rules of subset having higher confidence 
    4) Sort the rules by decreasing lift
    
It is a good algorithm for reccomenders. 

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, len(dataset.values[i,:]))])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# min_support:      what support you expect at least. We one a product that at least was purchse 3/4 times a day 7daysperweek*3/7500
# min_confidence:   the number of times the rules need to be correct. Too high will lead to obvious results.
# min_lift:         We want rules above lift 3 -> to get relevant rules
# min_length : we want at least 2 products in the basket

# Visualising the results - rules sorted by own relevance
results = list(rules)