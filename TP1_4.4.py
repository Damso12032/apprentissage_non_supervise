


import numpy as np
import matplotlib.pyplot as plt
import time
import hdbscan

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
##################################################################
# Exemple : DBSCAN Clustering


path = './clustering-benchmark-master/src/main/resources/datasets/artificial/'
name="xor.arff"

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


min_cluster_size=12
###Tracé des clusters avec les paramétres optimaux trouvés
min_samples_elements=[i for i in range(2,15)]
silhouette_scores=[]
print("Appel HDBSCAN (1) ... ")
for min_samples in min_samples_elements:
   # min_samples=10
    tps1 = time.time()
    model = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette_scores.append(silhouette_score(datanp,labels))
plt.plot(min_samples_elements,silhouette_scores)
plt.xlabel("nb_cluster")
plt.ylabel("silhouette_score")
plt.title("silhouette_score=f(nb_cluster) with linkage=average")
plt.grid()
plt.show()


index_min_samples=silhouette_scores.index(max(silhouette_scores))
min_samples=min_samples_elements[index_min_samples]
model = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size)
model.fit(datanp)
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering HDBSCAN (1) - min_samples= "+str(min_samples)+" min_cluster_size= "+str(min_cluster_size))
plt.show()





'''
#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()
'''

# faire vairier epsilon et fixer min_pts afin d'avoir l'évolution du coefficient de silhouette en fonction de epsilon
# 
