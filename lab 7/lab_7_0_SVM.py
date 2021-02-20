# -*- coding: utf-8 -*-
"""
@author: ravros
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)

X = irisdata.drop('Class', axis=1)
y = irisdata['Class']
X = X[0:100]
y = y[0:100]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)



def change_lab(label):
    new_y = []
    count = 0
    dictionaryArr = dict()
    for key in label:
        if not (key in dictionaryArr):
            dictionaryArr[key] = dictionaryArr.get(key, count)
            count += 1
        val = dictionaryArr[key]
        new_y.append(val)
    return new_y


def classify_lb1(v, label):
    counter = 0
    res1_lis = []
    res2_lis_names = []
    count = 0
    dictionaryArr = dict()
    for key in label:
        if not (key in dictionaryArr):
            dictionaryArr[key] = dictionaryArr.get(key, count)
            res1_lis.append([])
            res2_lis_names.append(key)
            count += 1
        res1_lis[res2_lis_names.index(key)].append(v[label.index[counter]])
        counter += 1
    return res1_lis[0], res1_lis[1]
"""
def classify_lb(v, label):
    counter = 0
    res1_lis = []
    res2_lis = []
    for i in label:
        if i ==  'Iris-versicolor':
            res1_lis.append(v[label.index[counter]])
        else:
            res2_lis.append(v[label.index[counter]])
        counter += 1
    return res1_lis, res2_lis
"""

z = X_train['sepal-length']
y1, y2 = classify_lb1(z, y_train) 
w = X_train['sepal-width']    
y3, y4 = classify_lb1(w, y_train)


new_y = change_lab(y_train)
plt.plot()
plt.subplot()
plt.title('Dataset')
plt.ylabel('sepal-width (cm)')
plt.xlabel('sepal-length (cm)')
plt.scatter(z, w, label = 'Iris-setosa',color='purple')
plt.scatter(z, w, label = 'Iris-versicolor',color='yellow')
plt.scatter(z, w, c=new_y,cmap='viridis')
plt.legend(loc='upper right');
plt.show()


v = X_test['sepal-length']
x1, x2 = classify_lb1(v, y_test)
m = X_test['sepal-width']
x3, x4 = classify_lb1(m, y_test)

 

plt.plot()
plt.subplot()
plt.title('Dataset')
plt.xlabel('sepal-length (cm)')
plt.ylabel('sepal-width (cm)')
plt.scatter(y1, y3, label = 'Iris-setosa-train')#, marker='^'
plt.scatter(y2, y4, label = 'Iris-versicolor-train')
plt.scatter(x1, x3, label = 'Iris-versicolor-test')
plt.scatter(x2, x4, label = 'Iris-setosa-test')
plt.scatter(x2, x4, marker='o',facecolors='none' ,edgecolors='red' ,s=300)
plt.scatter(x1, x3, marker='x',linewidths=0.1 ,s=300, color='red')
plt.legend(loc='upper right');
plt.show()



def calcRates(TP, TN, FP, FN):
    TPR = TP/(TP+FN)
    FPR = 1-(TN/(TN+FP))
    ACC = (TP+TN)/(TP+TN+FP+FN)
    PREC = TP/(TP+FP)
    return (TPR, FPR, ACC, PREC)



res1 = change_lab(y_test)
res2 = change_lab(y_pred)

tp=tn=fp=fn=0
for i in range(0,len(res2)):
    if res2[i]==res1[i]:
        if res2[i]==0:
            tp+=1
        else:
            tn+=1
    else:
        if res2[i]==1:
            fp+=1
        else:
            fn+=1
print(tp, tn, fp, fn)
my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(my_res[0]))
print("False Positive Rate "+str(my_res[1]))
print("Accuracy "+str(my_res[2]))
print("Precision "+str(my_res[3])) 





new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X, y, test_size = 0.90)
new_svclassifier = SVC(kernel='poly', degree=8)
new_svclassifier.fit(new_X_train, new_y_train)
new_y_pred = new_svclassifier.predict(new_X_test)


z = new_X_train['sepal-length']
y1, y2 = classify_lb1(z, new_y_train) 
w = new_X_train['sepal-width']    
y3, y4 = classify_lb1(w, new_y_train)


new_y = change_lab(new_y_train)
plt.plot()
plt.subplot()
plt.title('Dataset')
plt.ylabel('sepal-width (cm)')
plt.xlabel('sepal-length (cm)')
plt.scatter(z, w, label = 'Iris-setosa',color='purple')
plt.scatter(z, w, label = 'Iris-versicolor',color='yellow')
plt.scatter(z, w, c=new_y,cmap='viridis')
plt.legend(loc='upper right');
plt.show()


v = new_X_test['sepal-length']
x1, x2 = classify_lb1(v, new_y_test)
m = new_X_test['sepal-width']
x3, x4 = classify_lb1(m, new_y_test)

 

plt.plot()
plt.subplot()
plt.title('Dataset')
plt.xlabel('sepal-length (cm)')
plt.ylabel('sepal-width (cm)')
plt.scatter(y1, y3, label = 'Iris-setosa-train')#, marker='^'
plt.scatter(y2, y4, label = 'Iris-versicolor-train')
plt.scatter(x1, x3, label = 'Iris-versicolor-test')
plt.scatter(x2, x4, label = 'Iris-setosa-test')
plt.scatter(x2, x4, marker='o',facecolors='none' ,edgecolors='red' ,s=300)
plt.scatter(x1, x3, marker='x',linewidths=0.1 ,s=300, color='red')
plt.legend(loc='upper right');
plt.show()


new_res1 = change_lab(new_y_test)
new_res2 = change_lab(new_y_pred)

tp=tn=fp=fn=0
for i in range(0,len(new_res2)):
    if new_res2[i]==new_res1[i]:
        if new_res2[i]==0:
            tp+=1
        else:
            tn+=1
    else:
        if new_res2[i]==1:
            fp+=1
        else:
            fn+=1
print(tp, tn, fp, fn)
new_my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(new_my_res[0]))
print("False Positive Rate "+str(new_my_res[1]))
print("Accuracy "+str(new_my_res[2]))
print("Precision "+str(new_my_res[3])) 



def classify_lb2(v, label):
    counter = 0
    res1_lis = []
    res2_lis_names = []
    count = 0
    dictionaryArr = dict()
    for key in label:
        if not (key in dictionaryArr):
            dictionaryArr[key] = dictionaryArr.get(key, count)
            res1_lis.append([])
            res2_lis_names.append(key)
            count += 1
        res1_lis[res2_lis_names.index(key)].append(v[counter])
        counter += 1
    return res1_lis[0], res1_lis[1]




from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=200, centers=2, cluster_std = 0.20, random_state = 5)
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size = 0.20)
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

z = X_train[:,0]
y1, y2 = classify_lb2(z, y_train) 
w = X_train[:,1]   
y3, y4 = classify_lb2(w, y_train)


new_y = y_train
plt.plot()
plt.subplot()
plt.title('Dataset')
plt.ylabel('attribute 1')
plt.xlabel('attribute 2')
plt.scatter(z, w, label = 'Group-1',color='purple')
plt.scatter(z, w, label = 'Group-2',color='yellow')
plt.scatter(z, w, c=new_y,cmap='viridis')
plt.legend(loc='upper right');
plt.show()


v = X_test[:,0]
x1, x2 = classify_lb2(v, y_test)
m = X_test[:,1]
x3, x4 = classify_lb2(m, y_test)

 

plt.plot()
plt.subplot()
plt.title('Dataset')
plt.ylabel('attribute 1')
plt.xlabel('attribute 2')
plt.scatter(y1, y3, label = 'Group-1-train')#, marker='^'
plt.scatter(y2, y4, label = 'Group-2-train')
plt.scatter(x1, x3, label = 'Group-2-test')
plt.scatter(x2, x4, label = 'Group-1-test')
plt.scatter(x2, x4, marker='o',facecolors='none' ,edgecolors='red' ,s=300)
plt.scatter(x1, x3, marker='x',linewidths=0.1 ,s=300, color='red')
plt.legend(loc='upper right');
plt.show()


new_res1 = y_test
new_res2 = y_pred

tp=tn=fp=fn=0
for i in range(0,len(new_res2)):
    if new_res2[i]==new_res1[i]:
        if new_res2[i]==0:
            tp+=1
        else:
            tn+=1
    else:
        if new_res2[i]==1:
            fp+=1
        else:
            fn+=1
print(tp, tn, fp, fn)
new_my_res = calcRates(tp, tn, fp, fn)
print("True Positive Rate (TPR) or Sensitivity "+str(new_my_res[0]))
print("False Positive Rate "+str(new_my_res[1]))
print("Accuracy "+str(new_my_res[2]))
print("Precision "+str(new_my_res[3])) 
