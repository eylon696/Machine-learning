# -*- coding: utf-8 -*-
"""
Image Processing
"""
#Import required libraries
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
from PIL import Image

#Open Image
im0 = Image.open("sad_cat.jpg")
title0 = Image.open("titles.jpg")
im0.show()
title0.show()

img = mpimg.imread('sad_cat.jpg')
title = mpimg.imread('titles.jpg')
title1 = title[:,:,0]

rows = title1.shape[0]
cols = title1.shape[1]

plt.plot()
plt.imshow(img, cmap = 'gray')

plt.plot()
plt.imshow(title, cmap = 'gray')

gr_im =  0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]

plt.plot()
plt.imshow(title1, cmap = 'gray')

plt.imshow(gr_im, cmap = 'gray')

title_inv = 255 - title1
#plt.imshow(title_inv, cmap = 'gray')

rows2=rows//2
title2 = title_inv[0:rows2,:]
#plt.imshow(title2, cmap = 'gray')

# find the start and end rows of "so fun" title 
title2 = np.array(title2)
counter = 0
flag = 1
for row in title2:
    for i in row:
        if i != 0 and flag:
            flag=0
            start = counter
        if i != 0:
            end = counter
    counter += 1
# print(start)
# print(end)

# find the start and end rows of "russian" title 
gr_im = np.array(gr_im)
counter = len(gr_im)
flag = 1
flag1 = 0
for row in  reversed(gr_im):
    for i in row:
        if i == 0 and flag:
            flag=0
            start_im = counter
        if i != 0:
            end_im = counter
            flag1 = 1
    if flag1:
        break
    counter -= 1
# print(start_im)
# print(end_im)


temp = end_im - (end - start)
# print(temp)
counter = 0
# "erase" the russian title 
for row in gr_im:
    for i in range(len(row)):
        if(counter>=temp):
            gr_im[counter, i] = 0
    counter+=1

# plt.imshow(gr_im, cmap = 'gray')

# find the start and end columns of "so fun" title 
counter = 0
starts = len(title2[0])
ends = 0
for i in range(start, end+1):
    for j in range(len(title2[0])):
        if title2[i, j] != 0:
            if j < starts:
                starts = j
            if j > ends:
                ends = j
# print(starts)
# print(ends)

#plt.imshow(gr_im, cmap = 'gray')


# "paste" so fun title onto the center of the "russian" title image
temp1 = start
temp2 = starts

my_range = ends - starts
value = len(gr_im[0])//2 - my_range//2
for i in range (temp, len(gr_im)):
    for j in range (value, value + my_range):
        gr_im[i,j]=title2[temp1, temp2]
        temp2 += 1
    temp1 += 1
    temp2 = starts
   
#plt.imshow(gr_im, cmap = 'gray')
