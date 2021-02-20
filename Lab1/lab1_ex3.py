# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:03:47 2020

@author: ראובן
"""

# import regular expressins packge
# import numbers package
import re
import numpy as np

def readFile(fileName):
    file = open(fileName,'r')
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
            

numFiles = 3
fileContent = ["","",""]

#read  and preprocess files 
fileContent[0] = preProcess(readFile('algebra.txt'))
fileContent[1] = preProcess(readFile('calculus.txt'))
fileContent[2] = preProcess(readFile('liter.txt'))

# Read dictionary file and make dictionary list
dictionary = readFile('dictionary.txt').split()

# count the number of dictionary words in files
frequency = np.empty((numFiles,len(dictionary)),dtype=np.int64)
for i in range(numFiles):
    for j,word in enumerate(dictionary):
        frequency[i,j] = len(re.findall(word,fileContent[i]))
        
# find the distance matrix between the text files
dist = np.empty((numFiles,numFiles))
for i in range(numFiles): 
    for j in range(numFiles):
        print(frequency[i,:]-frequency[j,:])
        # מחסרים כל איבר שבשורה i עם כל איבר שבשורה j התוצאה מטריצה בגודל 9 שורות על 15 עמודות
        # לאחר מכן np.linalg.norm מחשב מטריצה נורמלית ע"י Frobenius norm לוקחים כל איבר במטריצה מעלים אותו בריבוע מחברים את כל האיברים ולתוצאה עושים שורש לאחר מכן מאכסנים את התוצאה (המרחק) במטריצה חדשה
        dist[i,j] = np.linalg.norm(frequency[i,:]-frequency[j,:])
print("dist=\n",dist)        
        
    
 
    