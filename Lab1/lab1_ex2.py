# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:03:47 2020

@author: ראובן
"""

# import regular expressins packge
# import numbers package
import re
import numpy as np

numFiles = 3

fileContent = ["","",""] #list
#read first file
file = open('algebra.txt','r')
for line in file:
    fileContent[0] += line
    
#read second file
file = open('liter.txt','r')
for line in file:
    fileContent[1] += line     
#read third file
file = open('calculus.txt','r')
for line in file:
    fileContent[2] += line 

# Remove extra spaces
# Remove non-letter chars    
# Change to lower case
for i in range(0,numFiles):
    fileContent[i] = re.sub(" +"," ", fileContent[i])
    fileContent[i] = re.sub("[^a-zA-Z ]","", fileContent[i])
    fileContent[i] = fileContent[i].lower()

# Read dictionary file
dictionaryFile = open('dictionary.txt','r')
dictionaryContent = ""
for line in dictionaryFile:
    dictionaryContent += line

# make dictionary list    
dictionary = dictionaryContent.split()

# count the number of dictionary words in files
frequency = np.empty((numFiles,len(dictionary)))
for i in range(0,numFiles):
    for j,word in enumerate(dictionary):
        print(j,word,"\n")
        frequency[i,j] = len(re.findall(word,fileContent[i]))
        
# find the distance matrix between the text files
dist = np.empty((numFiles,numFiles))
for i in range(0,numFiles): 
    for j in range(0,numFiles):
        dist[i,j] = np.linalg.norm(frequency[i,:]-frequency[j,:])
        
print("dist=\n",dist)   
  
    
 
    