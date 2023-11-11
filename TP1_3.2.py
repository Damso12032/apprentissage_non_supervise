import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score

###################################################################
# Exemple : Agglomerative Clustering


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

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()


seuils_distances=[10+5*i for i in range(10)]
silhouette_scores=[]


### tracer silhouette_score=f(seuil_distance)
"""
for i in seuils_distances:
    tps1 = time.time()
    seuil_dist=i
    model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    silhouette_scores.append(silhouette_score(datanp,labels))
    k = model.n_clusters_
    leaves=model.n_leaves_
    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
    #plt.show()
    print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

#On fixe une méthode linkage et on fait varier la distance
plt.plot(seuils_distances,silhouette_scores)
plt.xlabel("seuil de distance")
plt.ylabel("silhouette_score")
plt.title("silhouette_score=f(seuil de distance) with average linkage")
plt.grid()
plt.show()
"""
### tracer silhouette_score=f(nb_cluster)

k_elements=[i for i in range(2,10)]
silhouette_scores=[]
times=[]
for k in k_elements:
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    model = model.fit(datanp)
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    silhouette_scores.append(silhouette_score(datanp,labels))
    leaves=model.n_leaves_
    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
    #plt.show()
    #print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

#On fixe une méthode linkage et on fait varier la distance
plt.plot(k_elements,silhouette_scores)
plt.xlabel("nb_cluster")
plt.ylabel("silhouette_score")
plt.title("silhouette_score=f(nb_cluster) with linkage=average")
plt.grid()
plt.show()




'''
###tracer temsp=f(linkage)
linkages=['average','average','single','complete']
for link in linkages:
    k=3average
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage=link, n_clusters=k)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    kres = model.n_clusters_
    leaves=model.n_leaves_
    #print(labels)
    #print(kres)average
    #silhouette_scores.append(silhouette_score(datanp,labels))
    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
    #plt.show()
    times.append(tps2-tps1)
    print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")
#On fixe une méthode linkage et on fait varier le nombre de clusters
plt.plot(linkages,times)
plt.xlabel("linkage")
plt.ylabel("duree")
plt.title("duree=f(linkage) with nb_cluster=3")
plt.grid()
plt.show()average

'''
##visualiser les clusters avec un k donné
k=3
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage="average", n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)
#silhouette_scores.append(silhouette_score(datanp,labels))
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
times.append(tps2-tps1)
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

###Calcul des scores de regroupement et séparation 
# Calcul de la somme des carrés des distances intra-cluster
intra_cluster_score = 0
for cluster_label in set(labels):
    cluster_points = datanp[labels == cluster_label]
    cluster_center = np.mean(cluster_points, axis=0)
    intra_cluster_score += np.sum((cluster_points - cluster_center) ** 2)

print("Score de Regroupement :", intra_cluster_score)

# Calcul des centres des clusters
cluster_centers = np.array([np.mean(datanp[labels == cluster_label], axis=0) for cluster_label in set(labels)])

# Initialisation du score de séparation
inter_cluster_score = 0

# Calcul de la somme des carrés des distances inter-cluster
for i, center1 in enumerate(cluster_centers):
    for j, center2 in enumerate(cluster_centers):
        if i != j:
            inter_cluster_score += np.sum((center1 - center2) ** 2)

inter_cluster_score /= (len(cluster_centers) * (len(cluster_centers) - 1))

print("Score de Séparation (Inter-Cluster) :", inter_cluster_score)