# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:35:44 2020

@author: ראובן
"""
# Text reading and writing
import re

#read file
file = open('algebra.txt','r')
fileContent = ""
# concat the read and concat the lines into string 
for line in file:
    fileContent += line        

# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
fileContent = re.sub(" +"," ", fileContent) #remove  extra spaces
fileContent = re.sub("[^a-z A-Z ]","", fileContent)
fileContent = fileContent.lower()
print( "file content ", fileContent)

