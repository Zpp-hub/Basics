# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:25:06 2019

@author: zhangpingping
"""
# =============================================================================
# 在图像上绘制各种图形
# =============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 画线
img = np.zeros((512, 512, 3), np.uint8)
cv2.line(img, (100, 200),(400, 300), (255, 0, 0), 5)
plt.imshow(img)

# 画矩形
cv2.rectangle(img, (384, 100), (200, 128), (0, 255, 0), 3)
plt.imshow(img)

# 画圆（圆心，半径）
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1) # -1是填充
plt.imshow(img)

# 椭圆（中心点，长轴，短轴，椭圆沿着顺时针的起始和结束角度）
cv2.ellipse(img, (256, 256), (100, 100), 0, 0, 100, 255, -1) # 第一个0 为起始角度
plt.imshow(img)

# 多边形(用数组画，并且每个类型为int32)
array_ = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
cv2.polylines(img, [array_], True, (255,255,0))
plt.imshow(img)

# 绘制实心多边形
cv2.fillConvexPoly(img, array_, (0,0,255)) # 绘制多边形
plt.imshow(img)

# 写字
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),10) # 字体大小，文字粗细
plt.imshow(img)




