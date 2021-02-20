"""
@author: katerina
"""
from math import sqrt
import numpy as np
#import datasets 
from sklearn import datasets
from pandas import DataFrame
import matplotlib.pyplot as plt 

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


#import datasets 
Iris = datasets.load_iris()

data = Iris.data[0:100,:]
lb = Iris.target[0:100] # true labeling

x1 = np.zeros((20,5))
x2 = np.zeros((20,5))
x1[0:20,0:4]=data[0:20,:]
x2[0:20,0:4]=data[50:70,:]
x1[0:20,4]=0
x2[0:20,4]=1
TrainingSet = np.concatenate((x1,x2),axis=0)

i=51
test_row=data[i,0:4]
for train_row in TrainingSet:
        min=100;
        dist = euclidean_distance(test_row[0:4], train_row[0:4])
        if dist<min:
           min=dist
           result=int(train_row[4])
print("The point", i, "is in the group", result)


y1 = np.zeros((30,5))
y2 = np.zeros((30,5))
y1[0:30,0:4]=data[20:50,:]
y2[0:30,0:4]=data[70:100,:]
y1[0:30,4]=3
y2[0:30,4]=4
Testing_set = np.concatenate((y1,y2),axis=0)

#def knn_alg(array,TrainingSet, k):
#    for elem in array:
        
#knn_alg(new_TrainingSet, TrainingSet, 1)

    
    
#def get_neighbors(Testing_set, TrainingSet):
 #   counter = 0 
 #   for new_row in Testing_set:
  #      min=60
   #     for row in TrainingSet:
    #        dist = euclidean_distance(new_row[0:4], row[0:4])
     #       if dist<min:
      #          min=dist
       #         Testing_set[counter,4]=int(row[4])
       # counter += 1
    #return Testing_set

#n = get_neighbors(Testing_set, TrainingSet)


def findKmin(test_row, TrainingSet, k):
    my_list_min = list()
    ret_my_list_min = list()
    for row in TrainingSet:
        dist = euclidean_distance(test_row[0:4], row[0:4])
        my_list_min.append((dist, int(row[4])))
    my_list_min.sort()
    for i in range(k):
        ret_my_list_min.append(my_list_min[i])
    return ret_my_list_min

def get_neighborsK(new_Testing_set, TrainingSet, k):
    counter = 0
    for test_row in new_Testing_set:
        dictionaryArr = dict()
        max_num = -1
        min_lis = findKmin(test_row, TrainingSet, k)
        for elem in min_lis:
            if not (elem[1] in dictionaryArr):
                dictionaryArr[elem[1]] = dictionaryArr.get(elem[1], 1)
            else:
                 temp = dictionaryArr[elem[1]]
                 dictionaryArr[elem[1]] = temp + 1
            for key,val in dictionaryArr.items():
                if val>max_num:
                    max_num = val
                    max_key = key      
        new_Testing_set[counter, 4] = max_key
        counter += 1
    #print(min_lis)
    #print(dictionaryArr)
    return new_Testing_set

temp = Testing_set.copy()
n = get_neighborsK(temp ,TrainingSet, 1)


group_one_x1 = x1[:,0]
group_one_x2 = x1[:,1]

group_two_x1 = x2[:,0]
group_two_x2 = x2[:,1]

group_one_n_x1 = n[0:30,0]
group_one_n_x2 = n[0:30,1]
group_two_n_x1 = n[30:60,0]
group_two_n_x2 = n[30:60,1]


plt.plot()
plt.subplot()
plt.title('Dataset')
plt.scatter(group_one_x1,group_one_x2,label = 'group 0')
plt.scatter(group_two_x1,group_two_x2, label = 'group 1')
plt.scatter(group_one_n_x1,group_one_n_x2, label = 'res group 0')
plt.scatter(group_two_n_x1,group_two_n_x2, label = 'res group 1')
plt.legend(loc='upper left');
plt.show()

x1 = Testing_set[:,0] 
x2 = Testing_set[:,1]
x3 = Testing_set[:,2]
x4 = Testing_set[:,3]


plt.plot()
plt.subplot()
plt.title('Dataset x1 and x2')
plt.scatter(x1,x2,c=Testing_set[:,4],cmap='viridis')
plt.show()

plt.plot()
plt.subplot()
plt.title('Dataset x3 and x4')
plt.scatter(x3,x4,c=Testing_set[:,4],cmap='viridis')
plt.show()
temp = Testing_set.copy()
n1 = get_neighborsK(temp ,TrainingSet, 3)


