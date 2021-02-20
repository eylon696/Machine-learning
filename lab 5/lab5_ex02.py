# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:36:13 2020

@author: ravros
"""

#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


# Generate some data
from pandas import DataFrame
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
X, y_true = make_blobs(n_samples=200, centers=2,
                       cluster_std = 0.90, random_state = 5)
X = X[:, ::-1] # flip axes for better plotting

# Plot the data with K Means Labels

kmeans = KMeans(2, random_state=2)
labels = kmeans.fit(X).predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');



def calcRates(TP, TN, FP, FN):
    TPR = TP/(TP+FN)
    FPR = 1-(TN/(TN+FP))
    ACC = (TP+TN)/(TP+TN+FP+FN)
    PREC = TP/(TP+FP)
    return (TPR, FPR, ACC, PREC)

def clust(df,n_cl):
    kmeans = KMeans(n_clusters = n_cl).fit(df)
    res = kmeans.labels_+1      
    centroids = kmeans.cluster_centers_

    plt.scatter(df['x1'], df['x2'], c = res, s=50, alpha = 1, cmap='viridis')

    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(res)),res)
    plt.show()
#show the silhouette values for k=3 
    silhouette_avg_ = silhouette_score(df, res)
    sample_silhouette_values_ = silhouette_samples(df, res)  
    plt.plot(sample_silhouette_values_) 
    plt.plot(silhouette_avg_, 'r--')
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    y=silhouette_avg_
    xmin=0
    xmax=len(res)
# The vertical line for average silhouette score of all the values
    plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    plt.show()

    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return res

y = np.array(X[:,0:2])
x1 = y[:,0] 
x2 = y[:,1]
df = DataFrame(y, columns=['x1','x2'])
res = clust(df, 2)
tp=tn=fp=fn=0
for i in range(0,len(y_true)):
    if y_true[i]==labels[i]:
        if y_true[i]==0:
            tp+=1
        else:
            tn+=1
    else:
        if y_true[i]==1:
            fp+=1
        else:
            fn+=1

my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(my_res[0]))
print("False Positive Rate "+str(my_res[1]))
print("Accuracy "+str(my_res[2]))
print("Precision "+str(my_res[3])) 
print(tp,tn,fp,fn)


new_X, new_y_true = make_blobs(n_samples=200, centers=2,
                       cluster_std = 0.80, random_state =2)
new_X = new_X[:, ::-1] # flip axes for better plotting

kmeans1 = KMeans(2, random_state=2)
labels1 = kmeans1.fit(new_X).predict(new_X)
#plt.scatter(new_X[:, 0], new_X[:, 1], c=labels1, s=40, cmap='viridis')
tp=tn=fp=fn=0
for i in range(0,len(new_y_true)):
    if new_y_true[i]==labels1[i]:
        if new_y_true[i]==0:
            tp+=1
        else:
            tn+=1
    else:
        if new_y_true[i]==1:
            fp+=1
        else:
            fn+=1

my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(my_res[0]))
print("False Positive Rate "+str(my_res[1]))
print("Accuracy "+str(my_res[2]))
print("Precision "+str(my_res[3])) 
print(tp,tn,fp,fn)
            




