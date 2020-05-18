# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:02:11 2019

@author: zpp
"""

class Student(object):
    """—__doc__打印的内容"""
    # 类变量
    CLASS_VAR = ['类变量和init同级']
    
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))
        

# 实例化       
class1 = Student('zpp','100')
class1.print_score()

# 打印类变量
class1.CLASS_VAR

# 打印文档信息
class1.__doc__
class1.__class__
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~继承的概念~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Parent():
    def __init__(self, last_name, eye_color):
        print('Parent Constructor')
        self.last_name = last_name
        self.eye_color = eye_color
    def show_info(self):
        print('Last Name - ' + self.last_name)
        print('Eye Color - ' + self.eye_color)
        
zpp = Parent('z', 'black') # 实例化
zpp.show_info()

# 继承子类(覆盖)
class Child(Parent):
    def __init__(self, last_name, eye_color, number_of_toys):
        print('Child Constructor')
        Parent.__init__(self, last_name, eye_color)
        self.number_of_toys = number_of_toys
    def show_info(self):
        print('Last Name - ' + self.last_name)
        print('Eye Color - ' + self.eye_color)
        print('Number Of Toys - ' + str(self.number_of_toys))        
        
zpp_child = Child('z', 'black', 5)
zpp_child.show_info()
print(zpp_child.number_of_toys)   
print(zpp_child.eye_color)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    