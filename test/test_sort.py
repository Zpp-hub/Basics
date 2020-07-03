# -*- coding: utf-8 -*-
import unittest
import random
from controller import sortController as soc


class Test_sort(unittest.TestCase):
    # before
    def setUp(self):
        print('Case Before')
        self.List = [16, 20, 50, 5, 1, 10]
        # self.List = [random.randrange(100) for i in range(10)]
        pass

    # after
    def tearDown(self):
        print('Case After')
        pass

    # 快速排序 先找第一个然后分左右 在递归
    def testQuickSort(self):
        rst = soc.quickSort(self.List)
        print(rst)

    # 冒泡排序
    def testBubbleSort(self):
        rst = soc.bubbleSort(self.List)
        print(rst)

    # 选择排序
    def testSelectSort(self):
        rst = soc.selectSort(self.List)
        print(rst)
