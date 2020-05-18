# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:52:03 2018

@author: win 10
"""
# python 基础 Series 和 DataFrame
# 加载库
import os
import numpy as np
import pandas as pd
#import time
# from datetime import datetime,timedelta
import decimal
import keyword

# 建立路径
os.chdir('G:\python_资料文件\机器学习唐宇迪\机器学习_自己\数据清洗')
print(os.getcwd())
# 查看保留字
print(keyword.kwlist)
input('name :')
print('name')

########################################### 一见钟情 #############################################
# A 字符串处理
text = 'database zpp'
text .count('a') # 计数
text.upper() # 大写
text.lower() # 小写
text.replace('a','b') # 替换
text.split(sep = 'a') # 分开
text.find('a') # 查找位置
print('{}叫{}{}'.format('我','啥','呢')) # 格式替换

# B 字符串切片
text[:3] # 0,1,2
text[4:6] # 4,5
text[:-1] # 删除后一个
text[::-2] # 从后往前取

# C 列表基本处理
list = [1,2,'ab',True,False]
len(list) # 长度
list * 2 # double list
list + list 
list.append('bb');list # 添加
list.insert(2,'dd');list # 按位置添加
list.pop(0);list
list.extend([1,2]);list # 增加元素
list.remove('cc');list
list.clear();list
list.reverse();list # 反排
list = [1,2,3]
list.sort();list


# D 列表索引
list[0] # 0
list[1:2] # 1
list[2:-2] # 2,3
list[-3] # 倒数第三个
list[2] = 'cc';list
list[3] = [3,2,1];list


# E 元组基础处理
tup = (1,2,31,65,5)
tup + tup
tup * 2

# F 字典常用操作
dac = {'aaa' : 25, 'bbb' : 32, 'ccc' : 63, 'dddd' : 65}
len(dac)
dac['bbb'] # 获取值
dac['bbb'] = 520;dac # 修改值
dac['new'] = 521;dac # 添加值
del dac['bbb']
dac.pop('aaa','no have') # 是否存在
dac.items()
dac.update({'k' : ' k1','m' : 'm1'});dac # 增加键值对
dac.update([('x',13),('x1',14)]);dac # 增加键值对
dac.setdefault('aaa',11)
dac.setdefault('lll',25) # 如果键不存在于字典中，将会添加键并将值设为默认值。
dac.get('aaa') # 得到对应的值
dac.keys() # 查看key
dac.values() # 查看值
dac.clear() # 清除

# G 集合基本操作 


# 创建Series  #s = pd.Series(data, index=index) 字典 ndarray 标量
all([1,1,0])

# H、时间格式
import time 
time.time()
time.localtime(time.time())
time.clock()
time.localtime(time.clock())
# 时间转字符串
a = time.localtime()
b = time.strftime('%Y-%m-%d %H:%M:%S',a)
c = time.strftime('%Y',a)
# 字符串转化为时间
a = '20180101 01:20:00'
b = time.strptime(a,'%Y%m%d %H:%M:%S')
c = time.mktime(b)
d = time.strftime('%Y-%m-%d %H:%M:%S',b)

import calendar
calendar.month(2017,7) # 日历

from datetime import datetime
now = datetime.now()
day = datetime(2018, 2, 3, 11 ,00)
# 转化为 timestamp = 0 = 1970-01-01 00:00:00
day.timestamp()
print(datetime.fromtimestamp(day.timestamp()))
# 时间转字符串
now = datetime.now()
print(now.strftime('%Y%m%d %H:%M:%S'))
# 字符转化时间
datetime.strptime('20130102 01:02:03','%Y%m%d %H:%M:%S')
print(datetime.strptime('20130102 01:02:03','%Y%m%d %H:%M:%S'))

# datetime加减 需要导入timedelta
from datetime import datetime, timedelta
now = datetime.now()
now + timedelta(days = 1,hours = 10)

######################################## 初步了解1 #########################################
###################################  pandas ####################################
############ Series
# 1、创建
# 字典
d = {'a' : 0, 'b' : 1, 'c' : 2}
pd.Series(d)
pd.Series(d,index = ['b','c','d','a'])

# ndarray
ser = pd.Series(np.random.randn(5), index = ['a','b','c','d','e'])
ser.index
ser = pd.Series(np.random.randn(5))
ser.index

# 标量
pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])
pd.Series(5)

# 2、 Series 切片
ser = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
ser
ser[0]
ser[:3]
ser[ser > 0.5]
ser[ser > ser.median()]
np.exp(ser)
ser = ser.append(pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']))
ser['a']
ser['e'] = 10
'e' in ser
print(ser.get('g'))
ser.get('g',np.nan)

# 3、矢量化操作
ser + ser
ser * 2
np.exp(ser)
ser[1:] + ser [:-1]

# 4、属性
s = pd.Series(np.random.randn(5), name='something')
s
s.name
s1 = s.rename('dif')
s1.name
s.index.name ='index_name'
s
s.values

####################### DataFrame 
# 1、创建
# 字典 + Series
d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df
pd.DataFrame(d, index=['d', 'b', 'a'])
pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
df.index
df.columns
 
# 字典 + ndarrays/lists
d = {'one' : [1., 2., 3., 4.],
     'two' : [4., 3., 2., 1.]}
pd.DataFrame(d)
pd.DataFrame(d, index=['a', 'b', 'c', 'd'])

# 其他
data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])
data[:] = [(1,2.,'Hello'), (2,3.,"World")]
data
pd.DataFrame(data, index=['first', 'second'])
pd.DataFrame(data, columns=['C', 'A', 'B'])

# 2、列操作
df['one']
df['three'] = df['one'] * df['two']
df['flag'] = df['one'] > 2
del df['two']
three = df.pop('three')
df['str'] = 'bar'
df['series'] = df['one'][:2]
df.insert(1, 'bar', df['one'])
# 索引 标签的索引.loc、位置的索引.iloc
df.loc['b'] 
df.loc['b','one'] 
df.iloc[1,1]
df.iloc[:,2]

df_sample = pd.DataFrame({'A': range(1, 11), 'B': np.random.randn(10)})
df_sample.assign(ln_A = lambda x: np.log(x.A), abs_B = lambda x: np.abs(x.B))

# 3、数据的对其运算
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
df + df2
df.iloc[0]
df - df.iloc[0]

index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
df.sub(df['A'], axis=0,level = int,fill_value = None)

df1 = pd.DataFrame({'a' : [1, 0, 1], 'b' : [0, 1, 1] }, dtype=bool)
df2 = pd.DataFrame({'a' : [0, 1, 1], 'b' : [1, 1, 0] }, dtype=bool)

df1 & df2
df1 | df2
df1 ^ df2 # 不同为真 相同为假
-df1  # ~df1
df[:5].T # 转置
np.exp(df)
np.asarray(df)

# 4、 数据IO
# 读取数据
# csv head tail
df = pd.read_csv("/WorkSpace/Data/TempData/webdata.csv")
df.head()
pd.read_table("/WorkSpace/Data/bikes.csv", sep=";", encoding="windows 1252", index_col=0, parse_dates=True).tail(10)
# pd.read_csv?

df_exc = pd.read_excel("/WorkSpace/Data/TempData/read_excel_demo.xlsx", sheetname="Variable Description", index_col=0)
df_exc.head(10)
# 读取数据库
import sqlite3
import mysql.connector
import sqlalchemy

mysql_engine = sqlalchemy.create_engine('mysql+mysqlconnector://root:1234@localhost/world', encoding='utf-8')
sql_table = pd.read_sql('show tables', mysql_engine)

sql_table
sqlite_engine = sqlalchemy.create_engine('sqlite:////WorkSpace/Data/SinaNews.db', encoding='utf-8')
sina_news = pd.read_sql("SELECT * FROM sinanewslink LIMIT 100", sqlite_engine)
sina_news.tail(10)

# 数据输出
df.to_csv("/WorkSpace/Data/TempData/test.csv", encoding="utf-8")
df.to_excel("/WorkSpace/Data/TempData/test.xlsx", "lookatme")
sqlite_engine_test = sqlalchemy.create_engine('sqlite:////WorkSpace/Data/TestDB.db', encoding='utf-8')
df.to_sql("dftest", sqlite_engine_test)
pd.read_sql("SELECT * FROM dftest", sqlite_engine_test, index_col="index").head(7)

################ 数据描述
index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
df
df.shape
df.index
df.columns
df.values
df.head()
df.tail(3)
df.info()
df.describe(percentiles = [*np.arange(0.2,1,0.2)])
df.T
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')


############### 选择&查询
# 获取列
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
                   'three' : pd.Series(np.random.randn(3), index=['b', 'c','d'])})
df.one
df["two"]

# 获取列
df[1:3]
df["a":"c"]
df['four'] = ['A', 'A','B','C']
df[df['four'].isin(['A','C'])] # 包含

# 按照标签索引.loc是基于标签的索引 df.loc[row,colum]
df2 = pd.DataFrame(np.random.randn(6,4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
df2 
df2.loc['a':'b'] # 行索引
df2.loc['a']
df2.loc[['a', 'b', 'd'], :]
df2.loc['d':, 'A':'C']
df2.loc[:, df2.loc['a'] > 0]
df2.loc['d','B']
df2.at['d','B']
# 赋值 
df2.loc[:,"E"] = df2.loc[:,"A"].apply(abs) # apply用法
df2.loc["g"] = 8.88
ser = df2.B
ser.loc['c':]
ser.loc['e']

# 基于位置索引 .iloc是基于位置的索引，传入的参数为目标区域的位置索引
ser = pd.Series(np.random.randn(5), index=list(range(0,10,2)))
ser.iloc[:3]
ser.iloc[3]
ser.iloc[:3] = 0 # 赋值

df4 = pd.DataFrame(np.random.randn(6,4),
                   index=list(range(0,12,2)),
                   columns=list(range(0,8,2)))
df4.iloc[:3]
df4.iloc[1:5, 2:4]
df4.iloc[[1, 3, 5], [1, 3]]
df4.iloc[1:3, :]
df4.iloc[:, 1:3]
df4.iloc[1, 1]
df4.iat[1,1] # 单个值索引效率高

# 使用函数作为参数
df5 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
df5
df5.loc[lambda df: df.A > 0, :]
df5.loc[:, lambda df: ['A', 'B']]
df5.iloc[:, lambda df: [0, 1]]
df5[lambda df: df.columns[0]]
df5.A.loc[lambda s: s > 0]

# 混合索引
df5.loc[df5.index[[0, 2]], 'A']
df5.iloc[[0, 2], df5.columns.get_loc('A')]
df5.iloc[[0, 2], df5.columns.get_indexer(['A', 'B'])]

# 布尔索引
s = pd.Series(range(-3, 4))
s[s > 0]
s[(s < -1) | (s > 0.5)]
s[~(s < 0)]
s[-(s < 0)]
df5[df5.A > 0]

# 结合map的复杂索引
df6 = pd.DataFrame({'a' : ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b' : ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c' : np.random.randn(7)})
df6
criterion = df6['a'].map(lambda x: x.startswith('t'))
df6[criterion]

df6[[x.startswith('t') for x in df6['a']]]
df6[criterion & (df6['b'] == 'x')]
df6.loc[criterion & (df6['b'] == 'x'),'b':'c']
# 结合isin 的复杂索引
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s.isin([2, 4, 6]) # 值包含
s[s.isin([2, 4, 6])]
s[s.index.isin([2, 4, 6])]
s[[2, 4, 6]]
  
df = pd.DataFrame({'vals': [1, 2, 3, 4],
                   'ids': ['a', 'b', 'f', 'n'],
                   'ids2': ['a', 'n', 'c', 'n']})
df
values = ['a', 'b', 1, 3]
df.isin(values)  
values = {'ids': ['a', 'b'], 'vals': [1, 3]}
df.isin(values)
  
# isin 与 any、all结合的复杂索引
values = {'ids': ['a', 'b'], 'ids2': ['a', 'c'], 'vals': [1, 3]}
row_mask = df.isin(values).all( axis = 1) # 行都为T 的
df[row_mask]

s[s > 0]
s.where(s > 0) # 全部保留不符合的为 NaN
df5[df5 < 0] # inplace可以用来修改原数据 
df5.where(df5 < 0, other=-df5) # 其他做处理

df6 = df5.copy()
df6[df6 < 0] = 0 # 查找即可替换

# 使用query()方法 # 视频了解
df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))
df[(df.a < df.b) & (df.b < df.c)]
df.query('(a < b) & (b < c)')
  
df = pd.DataFrame(np.random.randint(10 / 2, size=(10, 2)), columns=list('bc'))
df.index.name = 'a'
df  
df.query('a < b and b < c')
df.query('index < b < c')
  
#################################### 随机抽样 ######################################
# sample()
s = pd.Series([0,1,2,3,4,5])
s.sample()
s.sample(n = 3) 
s.sample(frac=0.5) # 百分比
s.sample(n=6, replace=False)
s.sample(n=7, replace=True) #  有放回
example_weights = [0, 0, 0.2, 0.2, 0.2, 0.4]
s.sample(n=3, weights=example_weights) # 权重
example_weights2 = [0.5, 0.1, 0, 0, 0, 0]
s.sample(n=2, weights=example_weights2)  
  
df2 = pd.DataFrame({'col1':[9,8,7,6], 'weight_column':[0.5, 0.4, 0.1, 0]})
df2  
df2.sample(n = 3, weights = 'weight_column')  
  
df3 = pd.DataFrame({'col1':[1,2,3], 'col2':[2,3,4]})
df3  
df3.sample(n=1, axis=1)  # 随机取列
df4 = pd.DataFrame({'col1':[1,2,3], 'col2':[2,3,4]})
df4.sample(n=2, random_state=2) # 随机种子

#axis、使用0值表示沿着每一列或行标签\索引值向下执行方法
#axis、使用1值表示沿着每一行或者列标签模向执行对应的方法
 
################################## 缺失值 ###########################################  
# 生成缺失值 
df = pd.DataFrame(np.random.randn(5, 3),
                  index=['a', 'c', 'e', 'f', 'h'],
                  columns=['one', 'two', 'three'])

df['four'] = 'bar'
df['five'] = df['one'] > 0
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
# 处理 isnull、notnull
pd.isnull(df2['one']) # 返回布尔值
df2.isnull()
df2['one'] == np.nan # np.nan 不是缺失值

# 插入缺失值
df3 = df[['one', 'two', 'three']].copy()
df3.iloc[[0,1,-1],0] = np.nan
df2 = df.copy()
df2['timestamp'] = pd.Timestamp('20120101')
df2.loc[['a','c','h'],['one','timestamp']] = np.nan
df2.get_dtype_counts() # 变量类型统计

# 缺失值的运算
#当进行求和运算时，缺失的数据将会被当作零来计算。
#如果数据全部为NA，那么结果也会返回NA。
#像 cumsum 和 cumprod 等方法回忽略NA值，但是会在返回的结果数组中回显示缺失的值。
# 在GroupBy中NA值会被直接忽略，这点同R相同
df3['one'].sum()
df3.mean(1)
df3.cumsum()
df3.groupby('one').mean()

# 清理填补缺失值 fillna
df2
df2.fillna(0)
df2['four'].fillna('missing')
# 向前填充
df2.fillna(method='bfill')
df2.fillna(method='bfill', limit=1)
df2.fillna(method='backfill')
df2.fillna(method='backfill', limit=1)
# 向后填充
df2.fillna(method = 'pad')
df2.fillna(method = 'ffill' ,limit=1)

# 用 pandas 对象填充
dff = pd.DataFrame(np.random.randn(10,3), columns=list('ABC'))
dff.iloc[3:5,0] = np.nan
dff.iloc[4:6,1] = np.nan
dff.iloc[5:8,2] = np.nan
dff
dff.fillna(dff.mean())
dff.fillna(dff.mean()['B':'C'])
dff.where(pd.notnull(dff), dff.mean(), axis='columns')

# 删除缺失值
# 在删除缺失值时需要注意的是，DataFrame有两条轴，而Series只有一条，所以需要指定在哪条轴上操作。
df3["one"] = np.nan
df3
df3.dropna(axis=0) # 行全部为nan的删除
df3.dropna(axis=1) # 列全部为nan的删除
df3['one'].dropna()

# 线性插值
ser = pd.Series([0.469112, np.nan, 5.689738, np.nan, 8.916232])
ser
ser.interpolate()

df = pd.DataFrame({'A': [1, 2.1, np.nan, 4.7, 5.6, 6.8],
                   'B': [.25, np.nan, np.nan, 4, 12.2, 14.4]})
df
df.interpolate() # 具体详见

# 替换
ser = pd.Series([0., 1., 2., 3., 4.])
ser.replace(0, 5)
ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
ser.replace({0: 10, 1: 100})
ser.replace([1, 2, 3], method='pad') # 替换成和上面一样的数

df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]})
df
df.replace({'a': 0, 'b': 5}, 100) # 两行同时替换一个值
d = {'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']}
df = pd.DataFrame(d)
df
df.replace('.', np.nan)
df.replace(r'\s*\.\s*', np.nan, regex=True)
df.replace(['a', '.'], ['b', np.nan]) # 两列替换成不同的值
df.replace([r'\.', r'(a)'], ['dot', '\1stuff'], regex=True)

################################## 分组计算 ###########################################  
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df
grouped = df.groupby('A')
grouped.groups
grouped = df.groupby(['A', 'B'])
grouped.groups

# 函数
def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel' # 元音
    else:
        return 'consonant' # 辅音
grouped = df.groupby(get_letter_type, axis=1)
grouped.groups

df3 = pd.DataFrame({'X' : ['A', 'B', 'A', 'B'], 'Y' : [1, 4, 3, 2]})
df3

df3 = pd.DataFrame({'X' : ['A', 'B', 'A', 'B'], 'Y' : [1, 4, 3, 2]})
df3
df3.groupby(['X']).get_group('A') # 取A组
df3.groupby(['X']).get_group('B') # 取B组
df
df.groupby('A').groups
df.groupby(['A']).max() # 诸多函数
df.groupby(['A', 'B']).groups
len(grouped)

grouped = df.groupby('A')
grouped.aggregate(np.sum) # 分组求和

grouped = df.groupby(['A', 'B'],as_index=False) # 作为列输出
grouped.aggregate(np.sum)
grouped.size() #查看分组大小
grouped = df.groupby('A')
grouped['C'].agg([np.sum, np.mean, np.std]) # 同时应用多个函数
grouped.agg([np.sum, np.mean, np.std]) # 对多列同时进行分组后几个函数的应用

grouped['C'].agg([np.sum, np.mean, np.std]).rename(columns={'sum': '1','mean': '2','std': '3'}) # 重命名
grouped.agg([np.sum, np.mean, np.std]).rename(columns={'sum': '1','mean': '2','std': '3'}) 
grouped.agg({'C' : np.sum,'D' : lambda x: np.std(x, ddof=1)}) # 对不同的列应用不同的函数


################################################## 初步了解 2 ###########################################
#################################### 数据合并 #########################################
# 连接
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                 'B': ['B0', 'B1', 'B2', 'B3'],
                 'C': ['C0', 'C1', 'C2', 'C3'],
                 'D': ['D0', 'D1', 'D2', 'D3']},
                 index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                 'B': ['B4', 'B5', 'B6', 'B7'],
                 'C': ['C4', 'C5', 'C6', 'C7'],
                 'D': ['D4', 'D5', 'D6', 'D7']},
                  index=[1 ,2 ,3 ,4 ])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                 'B': ['B8', 'B9', 'B10', 'B11'],
                 'C': ['C8', 'C9', 'C10', 'C11'],
                 'D': ['D8', 'D9', 'D10', 'D11']},
                 index=[8, 9, 10, 11])
df1
df2
df3
frames = [df1, df2, df3]
result = pd.concat(frames)
result
result = pd.concat(frames, keys=['x', 'y', 'z'])
result
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
              'D': ['D2', 'D3', 'D6', 'D7'],
              'F': ['F2', 'F3', 'F6', 'F7']},
             index=[2, 3, 6, 7])
pd.concat([df1, df4], axis=1)
pd.concat([df1, df4], axis=1, join='inner')
pd.concat([df1, df4], axis=1, join_axes=[df1.index])
df1.append(df2) # 增加行
df1.append(df4)
df1.append([df2, df3])
pd.concat([df1, df4], ignore_index=True)
df1.append(df4, ignore_index=True)
df2 = df2.reset_index(drop=True) # 重新生成从0开始的索引
pd.concat([df1, df2], axis=1)

s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
s1
pd.concat([df1, s1], axis=1)
s2 = pd.Series(['_0', '_1', '_2', '_3'])
s2
pd.concat([df1, s2, s2, s2], axis=1)
pd.concat([df1, s1], axis=1, ignore_index=True)
s2 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])
s2
df1.append(s2, ignore_index=True)

############# 数据库风格的DataFrame连接合并
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, on='key') # 有on忽视索引

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, how='left',on=['key1', 'key2']) # 相当于先合并在连接
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='outer', on=['key1', 'key2']) # 全连接

left = pd.DataFrame({'A' : [1,2], 'B' : [2, 2]})
right = pd.DataFrame({'A' : [4,5,6], 'B': [2,2,2]})
pd.merge(left, right, on='B', how='outer')

# 合并指示符
df1 = pd.DataFrame({'col1': [0, 1], 'col_left':['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2],'col_right':[2, 2, 2]})
pd.merge(df1, df2, on='col1', how='outer', indicator=True) # 是那个表存在的列出来
pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')

# 通过索引连接
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')
left.join(right, how='inner') # 交集
pd.merge(left, right, left_index=True, right_index=True, how='outer')
pd.merge(left, right, left_index=True, right_index=True, how='inner')

# 通过索引和列同时连接 (一个表用索引 一个表用列)
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})
right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                      index=['K0', 'K1'])
left.join(right, on='key') 
pd.merge(left, right, left_on='key', right_index=True, how='left', sort=False)

# 重叠名称的合并
left = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'v': [1, 2, 3]})
right = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'v': [4, 5, 6]})
result = pd.merge(left, right, on='k')
pd.merge(left, right, on='k', suffixes=['_l', '_r'])

# 数据框拼接
df1 = pd.DataFrame([[np.nan, 3., 5.], [-4.6, np.nan, np.nan],
                    [np.nan, 7., np.nan]])
df2 = pd.DataFrame([[-42.6, np.nan, -8.2], [-5., 1.6, 4]],
                    index=[1, 2])
df1.combine_first(df2)
df1.update(df2) # 将2中的值传入1

#################################### 数据重塑及透视表 #########################################
# 多重索引
columns = pd.MultiIndex.from_tuples([
        ('A', 'cat', 'long'), ('B', 'cat', 'long'),
        ('A', 'dog', 'short'), ('B', 'dog', 'short')],
         names=['exp', 'animal', 'hair_length'])
df = pd.DataFrame(np.random.randn(4, 4), columns=columns)
df
df.stack(level=['animal', 'hair_length']) # 横向转化为竖向
df.stack(level=[1, 2])

columns = pd.MultiIndex.from_tuples([('A', 'cat'), ('B', 'dog'),
                                     ('B', 'cat'), ('A', 'dog')],
                                    names=['exp', 'animal'])
index = pd.MultiIndex.from_product([('bar', 'baz', 'foo', 'qux'),
                                    ('one', 'two')],
                                   names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 4), index=index, columns=columns)
df2 = df.iloc[[0, 1, 2, 4, 5, 7]]
df2.stack('exp')
df2.stack('animal')
df3 = df.iloc[[0, 1, 4, 7], [1, 2]]
df3.unstack()
df3.unstack(fill_value=0)
df3.unstack(0)

############# 透视表
import datetime
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['A', 'B', 'C'] * 8,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                   'D': np.random.randn(24),
                   'E': np.random.randn(24),
                   'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +
                        [datetime.datetime(2013, i, 15) for i in range(1, 13)]})
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']) # 透视表
pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)
pd.pivot_table(df, values=['D','E'], index=['B'], columns=['A', 'C'], aggfunc=np.sum)
pd.pivot_table(df, index=['A', 'B'], columns=['C'])

############# 可视化
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.style.available)
matplotlib.style.use('seaborn')
%matplotlib inline

# 基础图
# 一列的 折线图
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
#plt.figure()
df.plot()
df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y='B')

# 条形图
df.iloc[5].plot(kind='bar')
plt.axhline(0, color='k')
df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot.bar()
df2.plot.bar(stacked=True) #堆叠的条形图
df2.plot.barh(stacked=True) # 水平的堆叠的条形图

# 直方图
df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
# plt.figure()
df4.plot.hist(alpha=0.5) # 正常
df4.plot.hist(stacked=True, bins=20) # 堆叠

# 箱线图
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()

color = dict(boxes='DarkGreen', whiskers='DarkOrange',
             medians='DarkBlue', caps='Gray')
df.plot.box(color=color, sym='r+')
df.plot.box(vert=False, positions=[1, 4, 5, 6, 8]) # 控制每个图的位置

df = pd.DataFrame(np.random.rand(10,5))
df.boxplot()

df = pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )
df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])
df.boxplot(by='X')

df = pd.DataFrame(np.random.rand(10,3), columns=['Col1', 'Col2', 'Col3'])
df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])
df['Y'] = pd.Series(['A','B','A','B','A','B','A','B','A','B'])
df.boxplot(column=['Col1','Col2'], by=['X','Y'])

# 面积图
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df.plot.area()
df.plot.area(stacked=False)

# 散点图
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')
ax = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1')
df.plot.scatter(x='c', y='d', color='r', label='Group 2', ax=ax)
df.plot.scatter(x='a', y='b', c='c', sharex=False, s=50)
df.plot.scatter(x='a', y='b', s=df['c']*200)

#  蜂巢图
df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)
df.plot.hexbin(x='a', y='b', sharex=False,  gridsize=25)

# 饼图
series = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')
series.plot.pie(figsize=(6, 6))
df = pd.DataFrame(3 * np.random.rand(4, 2), index=['a', 'b', 'c', 'd'], columns=['x', 'y'])
df.plot.pie(subplots=True, figsize=(8, 4))

series.plot.pie(labels=['AA', 'BB', 'CC', 'DD'], colors=['r', 'g', 'b', 'c'],
                autopct='%.2f', fontsize=20, figsize=(6, 6))

series = pd.Series([0.1] * 4, index=['a', 'b', 'c', 'd'], name='series2')
series.plot.pie(figsize=(6, 6))

# 缺失值的默认绘图方式
# 绘图工具
# 矩阵散点图
from pandas.plotting import scatter_matrix
df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

# 密度图
ser = pd.Series(np.random.randn(1000))
ser.plot.kde()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
df.plot(legend=False)
df.plot()

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = np.exp(ts.cumsum())
ts.plot(logy=True)

df.A.plot()
df.B.plot(secondary_y=True, style='g')

ax = df.plot(secondary_y=['A', 'B'])
ax.set_ylabel('CD scale')
ax.right_ax.set_ylabel('AB scale')































