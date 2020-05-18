sv# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:53:38 2019

@author: zhangpingping
"""
import cv2 
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
 
# 读取并显示
img = cv2.imread(r'E:\ZhiYuan\opencv\29443.fundus.png') 
cv2.namedWindow("Image") 
cv2.imshow("Image", img) 
cv2.waitKey (0)
cv2.destroyAllWindows()

# RGB三通道
emptyImage = np.zeros(img.shape, np.uint8)
emptyImage.shape
emptyImage2 = img.copy()
emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
emptyImage3[...]=0


# 展示和保存
cv2.imshow("EmptyImage", emptyImage)
cv2.imshow("Image", img)
cv2.imshow("EmptyImage2", emptyImage2)
cv2.imshow("EmptyImage3", emptyImage3)
cv2.imwrite("./cat2.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
cv2.imwrite("./cat3.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imwrite("./cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imwrite("./cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
cv2.waitKey (0)
cv2.destroyAllWindows()
 

# 访问像素
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'E:\ZhiYuan\opencv\29443.fundus.png')
n = 500
def salt(img, n):
	for k in range(n):
        # k = 0
		i = int(np.random.random() * img.shape[1]) # 行/宽
		j = int(np.random.random() * img.shape[0]) # 
		if img.ndim == 2: # 几维矩阵
			img[j,i] = 255
		elif img.ndim == 3: 
			img[j,i,0]= 0 # 蓝色B
			img[j,i,1]= 255 # 绿色G
			img[j,i,2]= 0 # 红色R
	return img

if __name__ == '__main__':
	img = cv2.imread(r'E:\ZhiYuan\opencv\29443.fundus.png')
	saltImage = salt(img, 500)
	cv2.imshow("Salt", saltImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# 分离、合并通道
import cv2
import numpy as np
 
img = cv2.imread(r'E:\ZhiYuan\opencv\_bscan.0.0.png')
b, g, r = cv2.split(img)
cv2.imshow("Blue", r)
cv2.imshow("Red", g)
cv2.imshow("Green", b)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 通道颜色和矩阵理解
emptyImage = np.zeros((496, 512), np.uint8) # 生成一个黑色的背景

# 更改背景颜色
rows,cols=emptyImage.shape # 496*512 
for i in range(rows):
    for j in range(cols):
            emptyImage[i,j]=0
            
cv2.namedWindow("Image") 
cv2.imshow("Image", emptyImage) 
cv2.waitKey (0)
cv2.destroyAllWindows()

# 生成一个椭圆
cv2.ellipse(emptyImage, (260, 240), (170, 130), 0, 0, 360, (255, 255, 255), 3)

cv2.namedWindow("Image") 
cv2.imshow("Image", emptyImage) 
cv2.waitKey (0)
cv2.destroyAllWindows()

# 检测轮廓
import numpy as np  
import cv2  
  
img = cv2.imread(r'E:\ZhiYuan\opencv\_bscan.0.0.png')  
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, thresh = cv2.threshold(imgray,127,255,0)  
contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)





