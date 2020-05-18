# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:59:02 2019

@author: zhangpingping
"""
# =============================================================================
# 几何变换：移动，旋转，仿射变换
# =============================================================================
# =============================================================================
# 放大或者缩小、指定大小
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("E:\ZhiYuan\opencv\duck.jpg")

# CV_INTER_NN - 最近邻插值,  
# CV_INTER_LINEAR - 双线性插值 (缺省使用)  
# CV_INTER_AREA - 使用象素关系重采样。

# 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
img.size
plt.imshow(img)
res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
plt.imshow(res)

# 这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子
height,width=img.shape[:2]
res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
plt.imshow(res)

# =============================================================================
# 将图像平移
# =============================================================================
cap=cv2.VideoCapture(0)
while(1):
# 获取每一帧
ret,frame=cap.read()
# 转换到 HSV
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 设定蓝色的阈值
lower_blue=np.array([10,50,50])
upper_blue=np.array([20,100,100])
# 根据阈值构建掩模
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# 对原图像和掩模进行位运算
res=cv2.bitwise_and(img, img, mask=mask)
# 显示图像
plt.imshow(img)
plt.imshow(mask)
plt.imshow(res)

# =============================================================================
# 图像旋转
# =============================================================================
img=cv2.imread("E:\ZhiYuan\opencv\duck.jpg",0)
rows,cols=img.shape
# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
# 第三个参数是输出图像的尺寸中心
dst=cv2.warpAffine(img,M,(2*cols,2*rows))
while(1):
    cv2.imshow('img',dst)
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()

# =============================================================================
# 仿射变换
# =============================================================================
img=cv2.imread('drawing.png')
rows,cols,ch=img.shape
pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])
M=cv2.getAffineTransform(pts1,pts2)
dst=cv2.warpAffine(img,M,(cols,rows))
plt.subplot(121,plt.imshow(img),plt.title('Input'))
plt.subplot(121,plt.imshow(img),plt.title('Output'))
plt.show()


# =============================================================================
# 透视变换
# =============================================================================
img=cv2.imread('sudokusmall.png')
rows,cols,ch=img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(300,300))
plt.subplot(121,plt.imshow(img),plt.title('Input'))
plt.subplot(121,plt.imshow(img),plt.title('Output'))
plt.show()






