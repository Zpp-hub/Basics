# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:16:50 2019

@author: zhangpingping
"""

# =============================================================================
# 视频的读取保存
# =============================================================================
# 摄像头捕获视频
import numpy as np
import cv2 

cap = cv2.VideoCapture(0) # 设备号

while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()

# 从文件中获取视频
import numpy as np
import cv2

cap = cv2.VideoCapture(0)













