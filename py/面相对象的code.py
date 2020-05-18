# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:20:44 2019

@author: zpp
"""

class Student(object):
    pass 

bart = Student()
bart # <__main__.Student at 0x1380e7f0048>;object/实例化的地址内存地址; 每个object/实例化的地址不同
bart1 = Student()
bart1
bart.name = 'aaa'
bart.name


class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score

bart = Student('Bart Simpson', 59)
bart.name

bart.score

def print_score(std):
    print('%s: %s' % (std.name, std.score))


class Student(object):
    """print doc_"""
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self):
        print('%s: %s' % (self.name, self.score))
class1 = Student('zpp','100')
class1.print_score()
class1.__doc__

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo")
args = parser.parse_args()
print (args.echo)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("echo", help="echo the string you use here")
args = parser.parse_args()
print (args.echo)
    
class Province():
    #静态字段（对象）
    country = '中国'
    def __init__(self, name):
        #普通字段(类)
        self.name = name
        
#直接访问普通字段
obj=Province('河北省')
print (obj.name)
#直接访问静态字段
Province.country
        
        
        
        
        
        
        
        
        
        
        
        