
# import regular expressins packge
# import numbers package

#from sklearn.datasets import make_blobs
from sklearn import cluster
#from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import numpy as np
import re

#import matplotlib.pyplot as plt 
#import pandas as pd 
#from pandas import DataFrame

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
            
rows = 5
fileContent = [""]*rows

#read  and preprocess files 
fileContent[0] = preProcess(readFile('DB.txt'))
fileContent[1] = preProcess(readFile('HP_small.txt'))
fileContent2 = preProcess(readFile('Tolkien.txt'))
numParts = 3
# split the third file to parts
partLength = int(len(fileContent2)/numParts) 
fileContent[2]  = fileContent2[0:partLength]
fileContent[3]  = fileContent2[partLength:partLength*2]
fileContent[4]  = fileContent2[partLength*2:partLength*3]
 
# concat files contents
numFiles = 5
allFilesStr = ""
for i in range(numFiles):
    allFilesStr += fileContent[i]

# generate a set of all words in files 
wordsSet =  set(allFilesStr.split())

# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)

# count the number of dictionary words in files
wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,fileContent[i]))
        
# find the distance matrix between the text files
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
        
print("dist=\n",dist)
        
# find the sum of the frequency colomns and select colomns having sum > 20
minSum = 20
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)
wordFrequency2 = np.empty((rows,len(indexArray[0])),dtype=np.int64)

for j in range(len(indexArray[0])):
    wordFrequency2[:,j] = wordFrequency[:,indexArray[0][j]]
    
dist2 = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist2[i,j] = np.linalg.norm(wordFrequency2[i,:]-wordFrequency2[j,:])
print("dist2=\n",dist2)


rows = 6
myFileContent = [""]*rows

numParts = 2
fileContentEliot = preProcess(readFile('Eliot.txt'))
# split the third file to parts
partLength = int(len(fileContentEliot)/numParts) 
myFileContent[0]  = fileContentEliot[0:partLength]
myFileContent[1]  = fileContentEliot[partLength:partLength*2]


numParts = 4
fileContentTolkien = preProcess(readFile('Tolkien.txt'))
# split the third file to parts
partLength = int(len(fileContentTolkien)/numParts) 
myFileContent[2]  = fileContentTolkien[0:partLength]
myFileContent[3]  = fileContentTolkien[partLength:partLength*2]
myFileContent[4]  = fileContentTolkien[partLength*2:partLength*3]
myFileContent[5]  = fileContentTolkien[partLength*3:partLength*4]

numFiles = 6
myAllFilesStr = ""
for i in range(numFiles):
    myAllFilesStr += myFileContent[i]

# generate a set of all words in files 
myWordsSet =  set(myAllFilesStr.split())

# Read stop words file - words that can be removed
myStopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
myDictionary = myWordsSet.difference(myStopWordsSet)

# count the number of dictionary words in files
wordFrequency3 = np.empty((rows,len(myDictionary)),dtype=np.int64)
for i in range(rows):
    for j,word in enumerate(myDictionary):
        wordFrequency3[i,j] = len(re.findall(word,myFileContent[i]))
        
# find the distance matrix between the text files
dist3 = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist3[i,j] = np.linalg.norm(wordFrequency3[i,:]-wordFrequency3[j,:])
print("dist3=\n",dist3)



# find the sum of the frequency colomns and select colomns having sum > 20
minSum = 20
rows = 6
mySumArray =  wordFrequency3.sum(axis=0)
myIndexArray = np.where(mySumArray > minSum)
wordFrequency4 = np.empty((rows,len(myIndexArray[0])),dtype=np.int64)

for j in range(len(myIndexArray[0])):
    wordFrequency4[:,j] = wordFrequency3[:,myIndexArray[0][j]]
    
dist4 = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist4[i,j] = np.linalg.norm(wordFrequency4[i,:]-wordFrequency4[j,:])
print("dist4=\n",dist4)
