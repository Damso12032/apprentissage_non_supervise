
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
name="tetra.arff"

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

# Distances aux k plus proches voisins
k = 6
neigh = NearestNeighbors ( n_neighbors = k )
neigh . fit ( datanp )
distances , indices = neigh . kneighbors ( datanp )
# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np . asarray ( [ np . average ( distances [ i ] [ 1 : ] ) for i in range (0 ,
distances . shape [ 0 ] ) ] )
# trier par ordre croissant
distancetrie = np . sort ( newDistances )
'''plt . title ( " Plus proches voisins " + str ( k ) )
plt . plot ( distancetrie ) 
plt.grid()
plt . show ()'''

###Tracé des distances moyennes sur les k plus proches voisins en spécifiant le coude


# Calcul des pentes entre les points successifs
slopes = np.diff(distancetrie)

# Trouver l'indice où la pente change de manière significative
elbow_index = np.argmax(slopes) +1  # Ajout de 1 car np.diff diminue la taille de l'array de 1

# Valeur correspondante sur l'axe x
elbow_value = distancetrie[elbow_index]

# Tracé du coude sur la courbe
plt.plot(distancetrie, marker='o')
plt.scatter(elbow_index, elbow_value, color='red', label='Coude')
plt.xlabel('Points')
plt.ylabel('Distances moyennes aux '+str(k)+' plus proches voisins')
plt.title('Courbe du coude')
plt.legend()
plt.grid()
plt.show()
print(f"Indice du coude : {elbow_index}, Valeur du coude : {elbow_value}")

###Tracé des clusters avec les paramétres optimaux trouvés

print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon=elbow_value
min_pts=k
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
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
'''
epsilons=[0.2*2*i for i in range(1,15)]
silhouette_scores=[]
for eps in epsilons:
    print("------------------------------------------------------")
    print("Appel DBSCAN (1) ... ")
    tps1 = time.time()

    min_pts= 8 #10   # 10

    epsilon=eps #2  # 4
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)
    silhouette_scores.append(silhouette_score(datanp,labels))

#plt.scatter(f0, f1, c=labels, s=8)
#on fixe min_pts à 5 et on fait varier epsilon
plt.plot(epsilons,silhouette_scores)
plt.title("silhouette_score=f(epsilon) avec min_pts=3")
plt.xlabel("epsilon")
plt.ylabel("silhouette_score")
plt.grid()
plt.show()


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
tps1 = time.time()
epsilon=0.05 #0.05
min_pts=5 # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()
'''



