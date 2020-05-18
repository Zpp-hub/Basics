# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:19:52 2019

@author: zhangpingping
"""

# =============================================================================
# 人机交互练习1
# =============================================================================
import cv2
dir(cv2)# 与help相近

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

drawing = False

# 如果mode为true绘制矩形，按下m键，变成绘制曲线
mode = True
ix, iy = -1,-1

# 创建回调函数
def drow_circel(event, x, y, flags, param):
    global ix,iy,drawing,mode

# 当按下左键是返回起始位置坐标
    

    

