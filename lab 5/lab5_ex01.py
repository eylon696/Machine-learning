"""
Created on Tue Oct 13 19:36:13 2020
@author: ravros
#IRIS DATA
"""
#import libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 
#import pandas as pd 
from pandas import DataFrame
from sklearn.metrics import silhouette_samples, silhouette_score
#import datasets 
from sklearn import datasets

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

#import datasets 
#from sklearn import datasets
Iris = datasets.load_iris()

y = Iris.data
lb = Iris.target # true labeling
y0 = np.array(y[:,0:4])
y2 = np.array(y[50:150:,0:4])
lb2 = np.array(lb[50:150])


x1 = y0[:,0] 
x2 = y0[:,1]
x3 = y0[:,2] 
x4 = y0[:,3]

df = DataFrame(y0, columns=['x1','x2','x3','x4']) 
plt.plot()
plt.subplot()
plt.title('Dataset')
plt.scatter(x1,x2,c=lb+1,cmap='viridis')
plt.show()

x1 = y2[:,0] 
x2 = y2[:,1]
x3 = y2[:,2] 
x4 = y2[:,3]

df1 = DataFrame(y2, columns=['x1','x2','x3','x4']) 

plt.plot()
plt.subplot()
plt.title('Dataset')
plt.scatter(x1,x2,c=lb2+1,cmap='viridis')
plt.show()

res3 = clust(df, 3)
res2 = clust(df1, 2)

# get data for TP FN FP TN
def get_data(arr, partition):
    arr_len = len(arr)
    arr_dict = dict()
    
    if (arr_len < partition):
        print("The array length is lower then the partition")
        return None
    
    for i in range(partition):
        key = arr[i]
        if not (key in arr_dict):
            arr_dict[key] = arr_dict.get(key, 0)
        arr_dict[key] += 1

    maximum = max(arr_dict.values())
    minimum = partition - maximum    
    return maximum, minimum

def calcRates(TP, TN, FP, FN):
    TPR = TP/(TP+FN)
    FPR = 1-(TN/(TN+FP))
    ACC = (TP+TN)/(TP+TN+FP+FN)
    PREC = TP/(TP+FP)
    return (TPR, FPR, ACC, PREC)

tp,fn = get_data(res2[0:50], 50)
tn,fp = get_data(res2[50:100], 50)
my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(my_res[0]))
print("False Positive Rate "+str(my_res[1]))
print("Accuracy "+str(my_res[2]))
print("Precision "+str(my_res[3])) 

   
