x=3
print(id(x))
x=5
print(id(x))
y=5
print(id(y))

print(3/5)
print(3//5)
print(4**0.5)

print([1,2,3]*2)#字符串也可
'''
x=int(input("input:"))#默认str型
print(x)
'''
for i in range(10,15):
    print(i,end=' ')#缩进即语法

import math
print(math.sqrt(4))


from math import*
print(sqrt(4))
'''
#pycharm download
#读入多个数据
a,b,c=map(int,input("3number:").split())#序列解包
print(a,b,c,type(a))
'''
#数字交换
a,b=map(int,input("a,b=").split())
if a>b:
    a,b=b,a
    print(a,b)
else:
    print("a<b")
a=6 if b>3 else 0
print(a)
#数组支持双向调用，a[-1]为最后一个元素,-2倒数第二个。。。
