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
name="xclara.arff"

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
inerties=[]
coefficients_silhouette=[]
times=[]
davies_bouldin_scores=[]
calinski_harabasz_scores=[]

# Run clustering method 
for k in range(2,10):
    print("------------------------------------------------------")
    print("Pour k=",k)
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

   

    print("nb_clusters trouvés:",max(labels)+1)
    classement=[[centroids[i]] for i in range(max(labels)+1)]
    for i in range(len(labels)):
        classement[labels[i]].append(datanp[i])
    #classement[i] contient tous les exemples qui sont dans le cluster i

    #scores_de_regroupement
    print("centre0:",centroids[0],"un elem:",classement[0][0])
    dist_min=[]
    dist_max=[]
    dist_av=[]
    for i in range(len(centroids)):
        minD=min(euclidean_distances(classement[i])[0][1:])
        maxD=max(euclidean_distances(classement[i])[0][1:])
        avgD=sum(euclidean_distances(classement[i])[0])/(len(classement[i])-1)
        dist_min.append(minD)
        dist_max.append(maxD)
        dist_av.append(avgD)
        print("Cluster ",i," : ")
        print("Distance minimale",minD)
        print("Distance maximale",maxD)
        print("Distance moyenne",avgD)

    #dist_min[i] est la distance minimale entre le cluster i et ses éléments 
    print("Distance minimale",dist_min)
    print("Distance maximale",dist_max)
    print("Distance moyenne",dist_av)

    #dist_min[i] est la distance minimale entre le cluster i et ses éléments 
    print("Distance minimale moyenne",sum(dist_min)/len(dist_min))
    print("Distance maximale moyenne",sum(dist_max)/len(dist_max))
    print("Distance moyenne moyenne",sum(dist_av)/len(dist_av))

    #scores de separation
    dist_av1=0
    dist_points=[]
    cmp=0
    for i in range(len(centroids)):
        for j in range(0,i):
            dist_points.append(euclidean_distances(centroids)[i][j])
            dist_av1+=euclidean_distances(centroids)[i][j]
            cmp+=1
    dist_max1=max(dist_points)
    dist_min1=min(dist_points)
    dist_av1=dist_av1/cmp
    print("Separation minimale",dist_min1)
    print("Separation maximale",dist_max1)
    print("Separaton moyenne",dist_av1)
    k_elmts.append(k)
    inerties.append(inertie)
    coefficients_silhouette.append(silhouette_score(datanp,labels))
    times.append(tps2-tps1)
    davies_bouldin_scores.append(davies_bouldin_score(datanp,labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(datanp,labels))

print("inerties=",inerties)
print("coefficients_silhouette=",coefficients_silhouette)
print("k_elmts=",k_elmts)

###tracé de inertie=f(k)

plt.plot(k_elmts,inerties)
plt.xlabel("k")
plt.ylabel("Inertie")
plt.title("inertie=f(k)")
plt.grid()
plt.show()

###tracé de coefficients_silhouette=f(k)

plt.plot(k_elmts,coefficients_silhouette)
plt.xlabel("k")
plt.ylabel("coefficients_silhouette")
plt.title("coefficients_silhouette=f(k)")
plt.grid()
plt.show()

###tracé de davies_bouldin_scores=f(k)

plt.plot(k_elmts,davies_bouldin_scores)
plt.xlabel("k")
plt.ylabel("davies_bouldin_scores")
plt.title("davies_bouldin_scores=f(k)")
plt.grid()
plt.show()

###tracé de calinski_harabasz_scores=f(k)

plt.plot(k_elmts,calinski_harabasz_scores)
plt.xlabel("k")
plt.ylabel("calinski_harabasz_scores")
plt.title("calinski_harabasz_scores=f(k)")
plt.grid()
plt.show()

###Tracé des clusters pour le k optimal obtenu avec le tracé du coefficient de silhouette
index_k_opt=coefficients_silhouette.index(max(coefficients_silhouette))
k_opt=k_elmts[index_k_opt] 
model = cluster.MiniBatchKMeans(n_clusters=k_opt, init='k-means++', n_init=1)
model.fit(datanp)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()


    










