[TOC]

# 2021.7.22

## `argparse`--- 命令行选项

### 创建一个解析器

使用 [`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) 的第一步是创建一个 [`ArgumentParser`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser) 对象：

```python
    parser = argparse.ArgumentParser( description=__doc__ )
    
```

[`ArgumentParser`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser) 对象包含将命令行解析成 Python 数据类型所需的全部信息。

### 添加参数

给一个 [`ArgumentParser`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser) 添加程序参数信息是通过调用 [`add_argument()`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser.add_argument) 方法完成的。通常，这些调用指定 [`ArgumentParser`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser) 如何获取命令行字符串并将其转换为对象。这些信息在 [`parse_args()`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser.parse_args) 调用时被存储和使用。例如：

```python
# 训练设备类型
parser.add_argument('--device', default='cuda:0', help='device')
# 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset')
    
```

### 解析参数

[`ArgumentParser`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser) 通过 [`parse_args()`](https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser.parse_args) 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。

```python
args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

```

## `os.path`--- 常用路径操作

```python
>>>import os

>>>path = '/root/runoob.txt'
>>>print( os.path.basename(path) ) # 返回文件名
runoob.txt
>>>print( os.path.dirname(path) )    # 返回目录路径
/root
>>>print( os.path.split(path) )      # 分割文件名与路径
('/root', 'runoob.txt')
>>>print( os.path.join('root','test','runoob.txt') )  # 将目录和文件名合成一个路径
root/test/runoob.txt
>>>print(os.path.exists(path) )
True

```

## 读取txt每一行的内容

```python
#以读取train.txt得到xml_list为例子
with open(txt_path) as read:
	self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             		for line in read.readlines()] 
    # line得到每行，最后带一个换行符，line.strip()去掉换行符
    # str.strip() ,在 string 上执行 lstrip()和 rstrip()
    
```

## zip()

`zip()` 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]

```

## enumerate()

`enumerate()` 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
>>>seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
... 
0 one
1 two
2 three

```

## Python 函数装饰器

装饰器(Decorators)是修改其他函数的功能的函数。

- 首先，在python中我们可以定义另一个函数

```python
def hi(name="yasoob"):
    def greet():
        return "now you are in the greet() function"
    def welcome():
        return "now you are in the welcome() function"
hi()

```

​	调用hi(), greet()和welcome()将会同时被调用，且greet()和welcome()函数在hi()函数之外是不能访问的

- 假设用 funA() 函数装饰器去装饰 funB() 函数：

```python
#funA 作为装饰器函数
def funA(fn):
    #...
    fn() # 执行传入的fn参数
    #...
    return '...'

@funA
def funB():
    #...
    
```

等价于：

```python
def funA(fn):
    #...
    fn() # 执行传入的fn参数
    #...
    return '...'

def funB():
    #...

funB = funA(funB)

```

使用函数装饰器 A() 去装饰另一个函数 B()，其底层执行了如下 2 步操作：

1. 将 B 作为参数传给 A() 函数；
2. 将 A() 函数执行完成的返回值反馈回 B。

举个例子：

```python
#funA 作为装饰器函数
def funA(fn):
    print("C语言中文网")
    fn() # 执行传入的fn参数
    print("http://c.biancheng.net")
    return "装饰器函数的返回值"

@funA
def funB():
    print("学习 Python")
    
print(funB)

```

输出结果：

```
C语言中文网
学习 Python
http://c.biancheng.net
装饰器函数的返回值

```

可以观察到funB的返回值变了

实际上，所谓函数装饰器，就是通过装饰器函数，在不修改原函数的前提下，来对函数的功能进行合理的扩充。

如果funB带参数，可以在funA内部设计一个嵌套函数：

```python
def funA(fn):
    # 定义一个嵌套函数
    def say(arc):
        print("Python教程:",arc)
    return say

@funA
def funB(arc):
    print("funB():", a)
funB("http://c.biancheng.net/python")

```



# 2021.7.23

## 关于GPU

- CUDA out of memory —— 超GPU的内存了，应该减小`batch_size`的大小。
- `torch.cuda.is_available()` ：GPU是否可用
- `torch.cuda.device_count()`：返回gpu数量；
- `torch.cuda.get_device_name(0)`：返回gpu名字，设备索引默认从0开始；

## OrderedDict

python中的字典一般是无序的，因为它是按照hash来存储的，但是python中有个模块collections(英文，收集、集合)，里面自带了一个子类OrderedDict，实现了对字典对象中元素的排序。

```python
import collections
print("Regular dictionary")
d={}
d['a']='A'
d['b']='B'
d['c']='C'
for k,v in d.items():
    print(k,v)
print("Order dictionary")
d1 = collections.OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['1'] = '1'
d1['2'] = '2'
for k,v in d1.items():
    print(k,v)

输出：
Regular dictionary
a A
c C
b B

Order dictionary
a A
b B
c C
1 1
2 2

```

可以看到，同样是保存了ABC等几个元素，**但是使用OrderedDict会根据放入元素的先后顺序进行排序。**所以输出的值是排好序的。

OrderedDict对象的字典对象，**如果其顺序不同那么Python也会把他们当做是两个不同的对象**。

## Python项目文件引用问题

项目目录示意：

```
└── project
  ├── __init__.py
  ├── main.py
  └── modules
      ├── __init__.py
      └── module1.py
      └── module2.py
  └── ui
      ├── __init__.py
      └── view.py
      └── item.py
      
```

1. 首先，在顶层目录下，创建`__init__.py`文件，在各级包文件夹下也同时创建`__init__.py`文件；
2. `view.py`引用`item.py`内的函数或类，需采用如下方式：`from ui.item import test`。引用`modules`包下文件内的函数或类，需采用如下方式：

```python
from modules.module1 import crawl
from modules import module2

```

3. 项目目录下的`main.py`文件引用各个包下文件内的函数或类，需采用下面的方式：

```python
from ui.item import test
from modules.module1 import crawl
from modules import module2

```

可以发现，上述引用方式均为绝对引用，而不是下述相对引用方式



## `__init__.py`

`__init__.py` 文件的作用是将文件夹变为一个Python模块,Python 中的每个模块的包中，都有`__init__.py` 文件。



通常`__init__.py` 文件为空，但是我们还可以为它增加其他的功能。我们在导入一个包时，实际上是导入了它的`__init__.py`文件。这样我们可以在`__init__.py`文件中批量导入我们所需要的模块，而不再需要一个一个的导入。

```python
# package
# __init__.py
import re
import urllib
import sys
import os

# a.py
import package 
print(package.re, package.urllib, package.sys, package.os)

```

## torch.jit.is_scripting():





## 问题遗留：faster_rcnn_framework.py 255行





## torchvision.ops.MultiScaleRoIAlign



## isinstance() 

isinstance() 函数来判断一个对象是否是一个已知的类型。isinstance() 会认为子类是一种父类类型，考虑继承关系。

```python
isinstance(object, classinfo)

```

> - object -- 实例对象。
> - classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。

```python
isinstance(min_size, (list, tuple)) #是list和tuple中的一个就返回True

```

## torch.as_tensor()

```python
mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)

```

> - 如果data已经是一个tensor并且与返回的tensor具有相同的类型和相同的设备，那么不会发生复制，返回的tensor就是data，否则进行复制并返回一个新的tensor。且如果具有requires_grad=True，并保留计算图。
>
> - 相似的，如果data是一个相应dtype的ndarray，并且设备是cpu（numpy中的ndarray只能存在于cpu中），那么也不会进行任何复制，但是返回的是tensor，只是使用的内存相同。

## torch.nn.functional.interpolate

```python
    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True)[0]

```

> - input (Tensor) – 输入张量
>
> - scale_factor (float or Tuple[float]) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
>
> - mode (str) – 可使用的上采样算法，有’nearest’, ‘linear’, ‘bilinear’, ‘bicubic’ , ‘trilinear’和’area’. 默认使用’nearest’

## list拼接

> python合并list有几种方法：

# 2021.7.24

## tensor的某种计算

```python
>>> import torch
>>> a = torch.tensor([1,2,3])
>>> b = torch.tensor([4,5,6])
>>> a[:,None]
tensor([[1],
        [2],
        [3]])
# [r1, r2, r3]' * [s1, s2, s3]
>>> a[:,None]*b[None,:] # 扩展维度
tensor([[ 4,  5,  6],
        [ 8, 10, 12],
        [12, 15, 18]])

```

