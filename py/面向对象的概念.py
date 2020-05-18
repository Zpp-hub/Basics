# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:20:47 2019

@author: zpp
"""
class Student(object): # 类
    pass

bart = Student() # 指向的是一个实例

bart.name = 'zpp' # 属性
bart.name

# 在建立类的时候就强制固定一些类
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
bart = Student('zpp', '100')
bart.name

# 打印出类中的属性
def print_score(std):
    print('%s: %s' % (std.name, std.score))
print_score(bart)

# 可以将类和打印放在一起，称为类的方法
class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))


bart = Student('zpp', '100')
bart.print_score()

# 加强练习
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def get_grade(self):
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'
bart = Student('zpp', 100)
bart.get_grade()



