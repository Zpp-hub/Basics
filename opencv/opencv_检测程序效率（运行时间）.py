# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:21:55 2019

@author: zhangpingping
"""
# =============================================================================
# 检测程序效率
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
e1 = cv2.getTickCount()
image=cv2.imread(r"E:\ZhiYuan\opencv\sky.png")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
plt.imshow(image, cmap='gray')
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()

# 查看优化是否开启
cv2.useOptimized()
cv2.medianBlur()


x = 10
%timeit y=x**2
%timeit y=x**x
