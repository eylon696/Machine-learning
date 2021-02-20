# import regular expressins packge
# import numbers package
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

rows = 3
fileContent = [""]*rows

#read  and preprocess files 
fileContent[0] = preProcess(readFile('Eliot.txt'))
fileContent[1] = preProcess(readFile('Tolkien.txt'))
fileContent[2] = preProcess(readFile('DB.txt'))

#wind - chunks size 
wind = 20000
#Divide the each file into chunks of the size wind 
chunks1, count1 = partition_str(fileContent[0] , wind)
chunks2, count2 = partition_str(fileContent[1] , wind)
chunks3, count3 = partition_str(fileContent[2] , wind)

# Concatinate all te chunks
rows = count1 + count2 + count3
chunks = chunks1 + chunks2 + chunks3

# Construct dictionary lines 54 - 65 
# Concatinate files contents
numFiles = 3
allFilesStr = ""
for i in range(numFiles):
    allFilesStr += fileContent[i]

# Generate a set of all words in files 
wordsSet =  set(allFilesStr.split())

# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)

# Count the number of dictionary words in files - Frequency Matrix
wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    print(i)
    for j,word in enumerate(dictionary):
        
        wordFrequency[i,j] = len(re.findall(word,chunks[i]))
        
# find the distance matrix between the text files - Distance Matrix
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
        
# find the sum of the frequency colomns and select colomns having sum > 100
minSum = 100
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((rows,indexArraySize),dtype=np.int64)

# generate a frequencey file with the selected coloumns 
for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]

 # find the another distance matrix between the text files 
dist1 = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        dist1[i,j] = np.linalg.norm(wordFrequency1[i,:]-wordFrequency1[j,:])
   
np.save('dist2',dist1,allow_pickle = True)  
     