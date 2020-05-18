# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:29:24 2019

@author: zhangpingping
"""
# =============================================================================
# 获取并修改像素值
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg")

# 获取某个位置
px = img[500, 500]
blue = img[500, 500, 0]
green = img[500, 500, 1]
red = img[500, 500, 2]

# 改变值
img[500, 500] = [255, 255, 255]
print(img.item(50, 50, 2))
img.shape
img.size # 返回像素数目
img.dtype # 返回像素的数据类型

b, g, r = cv2.split(img) # 拆分通道
img = cv2.merge([b, g, r]) # 合并通道

# 图像扩边 好多种有需要再细看
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"E:\ZhiYuan\opencv\duck.jpg")
b,g,r = cv2.split(img)
img1 = cv2.merge([r,g,b])
plt.subplot(122);plt.imshow(img1) # expect true color

replicate = cv2.copyMakeBorder(img1,50,50,50,50,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,50,50,50,50,cv2.BORDER_REFLECT)
wrap = cv2.copyMakeBorder(img1,50,50,50,50,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,50,50,50,50,cv2.BORDER_CONSTANT)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()


