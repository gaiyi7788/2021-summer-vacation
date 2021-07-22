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

