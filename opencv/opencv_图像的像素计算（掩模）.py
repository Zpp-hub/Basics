# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:59:12 2019

@author: zhangpingping
"""
# =============================================================================
# 图像的算术运算
# =============================================================================
"""
普通的相加
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x, y))
print(x+y)

"""
两张图的混合
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1=cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg")
b,g,r = cv2.split(img1)
img1 = cv2.merge([r,g,b])

img2=cv2.imread(r"E:\ZhiYuan\opencv\duck.jpg")
b,g,r = cv2.split(img2)
img2 = cv2.merge([r,g,b])

img2=cv2.resize(img2,(400, 429),interpolation=cv2.INTER_CUBIC)
plt.imshow(img2)

dst=cv2.addWeighted(img1,0.5,img2,0.5,0)
plt.imshow(dst)

"""
按位进行运算
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# 加载图像
image=cv2.imread(r"E:\ZhiYuan\opencv\sky.png")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
plt.imshow(image, cmap='gray')
 
# 创建矩形区域，填充白色255
rectangle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), (255,255,255), -1) # (25, 25):左上点的坐标，矩阵的宽和高，255：rgb颜色,-1:是线宽
plt.imshow(rectangle, cmap='gray')
 
# 创建圆形区域，填充白色255
circle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.circle(circle, (150, 150), 150, (255,255,255), -1) # 圆点，半径，rgb颜色，线宽
plt.imshow(circle, cmap='gray')
 
 
# 在此例（二值图像）中，以下的0表示黑色像素值0, 1表示白色像素值255
# 位与运算，与常识相同，有0则为0, 均无0则为1
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
plt.imshow(bitwiseAnd, cmap='gray')

 
# 非运算，非0为1, 非1为0
bitwiseNot = cv2.bitwise_not(circle)
plt.imshow(bitwiseNot, cmap='gray')
 
# 或运算，有1则为1, 全为0则为0
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)
 
# 异或运算，不同为1, 相同为0
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
plt.imshow(bitwiseXor, cmap='gray')

"""
mask:掩膜操作
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# 加载图像
image=cv2.imread(r"E:\ZhiYuan\opencv\sky.png")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
plt.imshow(image, cmap='gray')
 
# 创建矩形区域，填充白色255
rectangle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.rectangle(rectangle, (360, 348), (660, 570), 255, -1) # (25, 25):左上点的坐标，矩阵的宽和高，255：rgb颜色,-1:是线宽
plt.imshow(rectangle, cmap='gray')
 
# 创建圆形区域，填充白色255
circle = np.zeros(image.shape[0:2], dtype="uint8")
cv2.circle(circle, (520, 455), 140, 255, -1) # 圆点，半径，rgb颜色，线宽
plt.imshow(circle, cmap='gray')

# 或运算，有1则为1, 全为0则为0
bitwiseOr = cv2.bitwise_or(rectangle, circle)
plt.imshow(bitwiseOr, cmap='gray')
# 使用mask
mask = bitwiseOr
plt.imshow(mask, cmap='gray')
 
# 就是mask中的0的位置不要，其他部位都要
masked = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(masked, cmap='gray')
