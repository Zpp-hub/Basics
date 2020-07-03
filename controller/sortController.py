# -*- coding: utf-8 -*-
import copy
# 快速排序
def quickSort(List):
    # 判断要排序的是否小于1
    if len(List) <= 1:
        return List
    left = []
    right = []
    for i in List[1:]:
        if i <= List[0]:
            left.append(i)
        else:
            right.append(i)
    return quickSort(left) + [List[0]] + quickSort(right)

# 冒泡排序
def bubbleSort(List1):
    List = copy.deepcopy(List1)
    print(List)
    for num in range(0, len(List)-1, 1):
        for i in range(0, len(List)-1-num, 1):
            if List[i] > List[i+1]:
                n = List[i+1]
                List[i + 1] = List[i]
                List[i] = n
        print(List)
    return List

# 选择排序
def selectSort(List1):
    List = copy.deepcopy(List1)
    for i in range(0, len(List), 1):
        min = List[i]
        index = i
        for j in range(i, len(List), 1):
            if ( List[j] < min):
                min = List[j]
                index = j
        n = List[i]
        List[i] = min
        List[index] = n
    return List