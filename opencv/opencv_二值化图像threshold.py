# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:41:29 2019

@author: zhangpingping
"""
# =============================================================================
# 二值化图像cv2.threshold, cv2.adaptiveThreshold
#cv2.threshold() :二值画图像（方法有如下几种）
#cv2.THRESH_BINARY： 黑白二值
#cv2.THRESH_BINARY_INV： 黑白二值反转
#cv2.THRESH_TRUNC: 大于阈值部分被置为threshold，小于部分保持原样  
#cv2.THRESH_TOZERO : 小于阈值部分被置为0，大于部分保持不变
#cv2.THRESH_TOZERO_INV:大于阈值部分被置为0，小于部分保持不变
# =============================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros([100, 100, 3], np.uint32)
img.size
a = np.array(np.arange(0,255, 2.55))
        

## 应用矩阵生成灰度图
a = np.arange(0,255, 2.55).reshape(1,100)
b = np.array(np.arange(0,255, 2.55).reshape(1,100))
for i in range(99):
    b = np.concatenate([b, a], axis = 0)
plt.imshow(b, cmap='gray')

c = np.reshape(b,(100, 100, 1))
d = np.concatenate([c,c],2)
e = np.concatenate([d,c],2)

e = np.array(e, dtype=np.uint32)
plt.imshow(e)

img = b
# 进行二值化处理
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# ret,thresh6 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

# 图像展示
titles = ['img','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# 图像展示
plt.subplot(231),plt.imshow(img,'gray') # 几行几列图中的第几个
plt.subplot(232),plt.imshow(thresh1,'gray')
plt.subplot(233),plt.imshow(thresh2,'gray')
plt.subplot(234),plt.imshow(thresh3,'gray')
plt.subplot(235),plt.imshow(thresh4,'gray')
plt.subplot(236),plt.imshow(thresh5,'gray')


"""
Otsu’s二值化
"""
img_bird = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg", 0)
plt.imshow(img_bird, cmap='gray')
#简单滤波
ret1,th1 = cv2.threshold(img_bird,127,255,cv2.THRESH_BINARY)
plt.imshow(th1, cmap='gray')

#Otsu 滤波
ret2,th2 = cv2.threshold(img_bird, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(ret2)
plt.imshow(th2, cmap='gray') 
 

# =============================================================================
# 自适应阈值
# 第一个原始图像
# 第二个像素值上限
# 第三个自适应方法Adaptive Method: 
#   — cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值 
#   —cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
# 第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
# 第五个Block size:规定领域大小（一个正方形的领域）
# 第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） 
# 这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
# =============================================================================
th2 = cv2.adaptiveThreshold(img_bird, 100, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)
 
