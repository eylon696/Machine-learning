# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:49:38 2020

@author: ravros
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score

#load file dist1.npy(from lab3_ex011.py)
dist2=np.load('dist2.npy')

#finction clust
def clust(dist,n_cl):
 
#cluster the data into k clusters, specify the k  
    kmeans = KMeans(n_clusters = n_cl)
    kmeans.fit(dist)
    labels = kmeans.labels_ + 1
#show the clustering results  
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(labels)),labels)
    plt.show()

# calculate the silhouette values  
    silhouette_avg_ = silhouette_score(dist, labels)
    sample_silhouette_values_ = silhouette_samples(dist, labels)
# show the silhouette values 
    plt.plot(sample_silhouette_values_) 
    plt.plot(silhouette_avg_, 'r--')
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    y=silhouette_avg_
    xmin=0
    xmax=len(labels)
# The vertical line for average silhouette score of all the values
    plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    plt.show()

    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return labels

#labels2 = clust(dist1, 2)
#labels3 = clust(dist1, 3)
labels2 = clust(dist2, 2)
labels3 = clust(dist2, 3)
labels3 = clust(dist2, 4)
