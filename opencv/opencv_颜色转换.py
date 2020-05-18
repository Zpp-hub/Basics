# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:11:17 2019

@author: zhangpingping
"""
# =============================================================================
# 颜色类型转化cv2.cvtColor()， cv2.inRange()
# =============================================================================
import cv2
"""
H（色彩/色度）的取值范围是 [0， 179]，
S（饱和度）的取值范围 [0， 255]， V（亮度）的取值范围 [0， 255]。但是不
同的软件使用的值可能不同。所以当你需要拿 OpenCV 的 HSV 值与其他软
件的 HSV 值进行对比时，一定要记得归一化
"""
# BGR-->Gray
input_image = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg")
Gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# BGR-->HSV
HSV_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)



### 物体跟踪
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while(1):
    # 获取每一帧
    ret,frame=cap.read()
    
    # 转换到 HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 设定蓝色的阈值
    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])
    
    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5)&0xFF
    if k==27:
        break
# 关闭窗口
cv2.destroyAllWindows()

