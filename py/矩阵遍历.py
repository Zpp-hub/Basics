# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:25:36 2019

@author: zhangpingping
"""
import numpy as np 
import cv2
import matplotlib.pyplot as plt
img = np.zeros((5, 4, 3), dtype=np.uint8)
img[2,0,0] = 255
img

points = np.array([[0,0], [2,0], [2,2], [0,2]], dtype=np.int32)
cv2.fillConvexPoly(img, points, [25, 25, 25])
plt.imshow(img)

img_0 = np.zeros((5, 5, 3), np.uint8)
points_0 = np.array([[1,1], [3,1], [3,3], [1,3]], dtype=np.int32)
cv2.fillConvexPoly(img_0, points_0, [25,25,25])
plt.imshow(img_0)

tt = [*map(lambda x, y: (x==255)&(y==255),img, img_0)]




