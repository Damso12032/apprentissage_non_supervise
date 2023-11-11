"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
##################################################################
# Exemple :  k-Means Clustering

path = './clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="square1.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne


k_elmts=[]
times=[]
times_mini=[]


# Variation de k et calcul des durées aprés avoir appliqué k-means et miniBatch
t0 = time.time()
for k in range(2,20):
    
    model2=cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    tps11=time.time()
    model2.fit(datanp)
    tps22=time.time()
    
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    tps1=time.time()
    model.fit(datanp)
    tps2=time.time()

    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

    k_elmts.append(k)
    times.append(tps2-tps1)
    times_mini.append(tps22-tps11)


tf = time.time()



plt.plot(k_elmts,times,color='blue',label="k-means")
plt.plot(k_elmts,times_mini,color='red',label="mini-batch")
plt.xlabel("K")
plt.ylabel("durée")
plt.legend()
plt.title("durée=f(k) pour square1")
plt.grid()
plt.show()














