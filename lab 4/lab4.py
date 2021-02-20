# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:30:27 2020

@author: Eylon
"""


import re
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score


#functions defonotion
def readFile(fileName):
    file = open(fileName,'r',encoding="cp437")
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr

# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr

#Divide the file in chuncks of the same size wind
def partition_str(fileStr, wind):
    n = wind
    chunks = [fileStr[i:i+n] for i in range(0, (len(fileStr)//n)*n, n)]
    #print(chunks)
    count = len(chunks)
    return chunks, count;


fileContent = [""]

#read  and preprocess files 
fileContent[0] = preProcess(readFile('text1.txt'))
#wind - chunks size 
wind = 50000
#Divide the each file into chunks of the size wind 
chunks1, count1 = partition_str(fileContent[0] , wind)

# Generate a set of all words in files 
wordsSet =  set(fileContent[0].split())

# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)

# Count the number of dictionary words in files - Frequency Matrix
wordFrequency = np.empty((count1,len(dictionary)),dtype=np.int64)
for i in range(count1):
    print(i)
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,chunks1[i]))

# find the sum of the frequency colomns and select colomns having sum > 100
minSum = 100
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((count1,indexArraySize),dtype=np.int64)

# generate a frequencey file with the selected coloumns 
for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]

dist1 = np.empty((count1,count1))
for i in range(count1): 
    for j in range(count1):
        dist1[i,j] = np.linalg.norm(wordFrequency1[i,:]-wordFrequency1[j,:])
   
np.save('dist1',dist1,allow_pickle = True) 


#load file dist1.npy(from lab4.py)
dist1=np.load('dist1.npy')

#function clust
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
    return labels, silhouette_avg_

max_sil = (-1, 0, 0)
for i in range(2, 5):
    labels ,sil = clust(dist1, i)
    if max_sil[0] < sil:
        max_sil = (sil, i, labels)

print("The number of books is: " + str(max_sil[1]))

def detectBoundaries(label):
    for i in range(0, len(label)-1):
        if label[i] != label[i+1]:
            end_book_1 = i
            start_book_2 = i + 1
            return end_book_1, start_book_2
    return -1, -1

end_book_1, start_book_2 = detectBoundaries(max_sil[2])
new_fileContent = [""]
new_fileContent[0] = chunks1[end_book_1]
new_fileContent[0] += chunks1[start_book_2]
new_wind = 5000
chunks2, count2 = partition_str(new_fileContent[0] , new_wind)

# Generate a set of all words in files 
new_wordsSet =  set(new_fileContent[0].split())

# Read stop words file - words that can be removed
new_stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
new_dictionary = wordsSet.difference(new_stopWordsSet)

# Count the number of dictionary words in files - Frequency Matrix
new_wordFrequency = np.empty((count2,len(new_dictionary)),dtype=np.int64)
for i in range(count2):
    print(i)
    for j,word in enumerate(new_dictionary):
        new_wordFrequency[i,j] = len(re.findall(word,chunks2[i]))

# find the sum of the frequency colomns and select colomns having sum > 100
minSum = 100
sumArray =  new_wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
new_wordFrequency1 = np.empty((count2,indexArraySize),dtype=np.int64)

# generate a frequencey file with the selected coloumns 
for j in range(indexArraySize):
    new_wordFrequency1[:,j] = new_wordFrequency[:,indexArray[0][j]]

dist2 = np.empty((count2,count2))
for i in range(count2): 
    for j in range(count2):
        dist2[i,j] = np.linalg.norm(new_wordFrequency1[i,:]-new_wordFrequency1[j,:])
   
np.save('dist2',dist2,allow_pickle = True) 

dist2=np.load('dist2.npy')
new_labels2, new_sil = clust(dist2, max_sil[1])


new_end_book_1, new_start_book_2 = detectBoundaries(new_labels2)
print("book 2 begins in the interval from "+ str(new_end_book_1*5000)+" to " + str(new_start_book_2*5000)+" in the clustered file")
print("book 2 begins in the interval from "+ str(end_book_1*wind +new_end_book_1*5000)+" to " + str(end_book_1*wind +new_start_book_2*5000)+" in the original file")
