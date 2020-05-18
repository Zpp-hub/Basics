# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:39:02 2019

@author: zhangpingping
"""
# =============================================================================
# 图片加载查看保存
# =============================================================================
import cv2
import numpy as np

img = cv2.imread(r"E:\ZhiYuan\opencv\相关公共图像\bird.jpg")# 加，0为灰度图
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"E:\ZhiYuan\opencv\bird_write.jpg", img)

# 按键操控
import cv2
import numpy as np

# 按下esc退出不保存，按下s退出并保存
img = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg", 0)
cv2.imshow('image', img)
k = cv2.waitKey()
if k ==27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite(r"E:\ZhiYuan\opencv\bird_write.jpg", img)
    cv2.destroyAllWindows()

#用matplotlib 实现相同的功能
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg", 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # 横纵坐标为空
plt.show()

## matplotlib 正常颜色显示读进来的图片
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"E:\ZhiYuan\opencv\bird.jpg")
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()

# cv2 显示
cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()

















