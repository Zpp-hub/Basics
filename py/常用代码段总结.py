# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:49:39 2019

@author: zhangpingping
"""
# =============================================================================
# 2019.04.24——2019.05.17
# =============================================================================
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pandas as pd
############################################################  lambda/map/apply/

list_ = [[1,1,3],[2,3,4],[3,2,5],[3,4,4]]
dataframe = pd.DataFrame(list_)

# lambda
var = lambda x: x**2
print(var(3))

var = lambda x, y, z : (x + y) ** z
print(var(1,2,2))

# map+lambda(可以多个参数)+for/if
map( lambda x: x*x, [y for y in range(10)])
[*map( lambda x: x*x, [y for y in range(10)])]

map( lambda x : x + 1, list_)
[*map( lambda x : x + 1, list_)]

[*map(lambda x, y: x if x>y else y,dataframe[0],dataframe[1])]

[*map(lambda x: x if x==2 else x-2 ,dataframe[0])]

# filter与map类似，但是结果只保留为True的项
[*filter(lambda x: x == 2,dataframe[0])]

# apply
dataframe.apply(lambda x :x.max(),axis = 0)

# 列表推导式
x = a if a>b else b
[exp for item in collection if condition]
[i for i in list_ if len(i)==3]

# 字典推导式
{key_exp:value_exp for item in collection if codition}
{key: value for key, value in enumerate(reversed(range(10)))}

# 集合推导式
{exp for item in collection if ondition}
{i for i in range(10)}

# 嵌套列表推导式
lists = [range(10), range(10, 20)]
[item for lst in lists for item in lst if item%2 == 0]


########################################################## cv2和numpy应用到的函数
# 生成一个背景为黑色的数组
img = np.zeros((5, 5, 3), dtype=np.uint8)
points = np.array([[0,0], [2,0], [2,2], [0,2]], dtype=np.int32) # 生成可以画在图像上的点
cv2.fillConvexPoly(img, points, [25, 25, 25]) # 画多边形
plt.imshow(img)

img_0 = np.zeros((5, 5, 3), np.uint8)
points_0 = np.array([[1,1], [3,1], [3,3], [1,3]], dtype=np.int32)
cv2.fillConvexPoly(img_0, points_0, [25,25,25])
plt.imshow(img_0)
tt = [*map(lambda x, y: (x==255)&(y==255),img, img_0)] # 交集
c = cv2.add(img, img_0)

############################################################## 字符串和list和数组
# 字符串
list_ = ['zhangpingping', '520', '+“+love+0-0+everyone']
str_ = 'zhangpingping+“+love+0-0+everyone'
str_.replace('+', '-', )
str_.count('+')
str_.split('+') # 字符串转化为列表
str_ = ''.join(list_) # 列表转化为字符串

# 字典转化为DataFrame
pd.DataFrame.from_dict(dict)
pd.concat([a,b], axis = 0)

#读取
with open('data.json', 'r', encoding='utf-8') as f:
    	data = json.load(f)
#写出
with open('data.json', 'w') as f: # f是位置及名字
    json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=3)




# E:\ZhiYuan\Task_2_cubiao\粗标数据示例 (1).json

with open(r'E:\ZhiYuan\Task_2_cubiao\json_in.json', 'r',encoding='utf-8') as f:
    	data = json.load(f)

with open(r'E:\ZhiYuan\Task_2_cubiao\json_out.json', 'w') as f: # f是位置及名字
    json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=3)






