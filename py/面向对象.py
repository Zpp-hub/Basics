# -*- coding: utf-8 -*-
# coding:utf-8
class Foo(object):
    """类的三种方法的语法形式"""

    # 实例化方法
    def instance_method(self):
        print('是类{}的实例化方法,只能被实例化调用'.format(Foo))

    @staticmethod
    def static_method():
        print('是静态方法')

    @classmethod
    def class_method(cls):
        print('类方法')

# 实例化
foo = Foo()
foo.instance_method()
foo.static_method()
foo.class_method()
print('------------------')
Foo.static_method()
Foo.class_method()

# 类方法（定义多个构造函数的情况）
class Book(object):
    def __init__(self, title):
        self.title = title

    @classmethod
    def create(cls, title):
        book = cls(title=title)
        return book

book1 = Book("python")
book2 = Book.create("python and django")
print(book1.title)
print(book2.title)

# 静态方法调用静态方法
class Foo(object):
    X = 1
    Y = 2

    @staticmethod
    def averag(*mixes):
        return sum(mixes) / len(mixes)

    @staticmethod
    def static_method():
        return Foo.averag(Foo.X, Foo.Y)

    @classmethod
    def class_method(cls):
        return cls.averag(cls.X, cls.Y)

foo = Foo()
print(foo.static_method())
print(foo.class_method())

# 继承类中的区别
class Foo(object):
    X = 1
    Y = 2

    @staticmethod
    def averag(*mixes):
        return sum(mixes) / len(mixes)

    @staticmethod
    def static_method():
        return Foo.averag(Foo.X, Foo.Y)

    @classmethod
    def class_method(cls):
        return cls.averag(cls.X, cls.Y)


class Son(Foo):
    X = 3
    Y = 5

    @staticmethod
    def averag(*mixes):
        return sum(mixes) / 3

p = Son()
print('----------------------------')
print(p.static_method())
print(p.class_method())

