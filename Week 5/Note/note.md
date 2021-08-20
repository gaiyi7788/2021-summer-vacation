[TOC]

# 2021.8.19

## Fluent python 第一章 Python 数据模型

### 魔术方法

python中特殊方法（魔术方法）是被python解释器调用的，我们自己不需要调用它们，也就是说没有 my_object.`__len__()` 这种写法我们统一使用内置函数来使用。`__len__()`和`__getitem__()`等都是特殊方法。特殊方法`__len__()`实现后，只需使用`len()`方法即可

- `__len__()`

一般返回数量，使用len()方法调用。在`len()`内部也可使用`len()`函数

- `__getitem__()`

此特殊方法一般是根据索引返回数据，也可以替代`__iter_()`和`__next__()`方法，也可支持切片

```python
# collections是Python内建的一个集合模块，提供了许多有用的集合类。
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])
class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]
    def __len__(self):
        return len(self._cards)
    def __getitem__(self, position):
        return self._cards[position]
```

因为 `__getitem__` 方法把 [] 操作交给了 self._cards 列表，所以我们的 deck 类自动支持切片（slicing）操作。

仅仅实现了 `__getitem__` 方法，deck 类就变成可迭代的了（替代了`iter()`）

- `__init__()`

定义`__init__`用于执行类的实例化的过程，`__init__`函数的参数列表会在开头多出一项，它永远指代新建的那个实例对象，习惯上就命为`self`。

- `__call__()`

实现后对象可变成可调用对象，此对象可以像函数一样调用，例如：自定义函数，内置函数，类都是可调用对象，可用callable()判断是否是可调用对象

```python
class Zarten():
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __call__(self):
        print('name:%s  age:%d' % (self.name, self.age))
z = Zarten('zarten', 18)
z()
>>name:zarten  age:18'
```

### Python sorted() 函数

**sorted()** 函数对所有可迭代的对象进行排序操作。

> **sort 与 sorted 区别：**
>
> sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
>
> list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。

```python
sorted(iterable, cmp=None, key=None, reverse=False)
```

> - iterable -- 可迭代对象。
> - cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
> - key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
> - reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

```python
>>> L=[('b',2),('a',1),('c',3),('d',4)]
>>> sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))   # 利用cmp函数
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]
>>> sorted(L, key=lambda x:x[1])               # 利用key
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]
```

### Python List index()方法

index() 函数用于从列表中找出某个值第一个匹配项的索引位置。

```
list.index(x, start, end)
```

> - x-- 查找的对象。
> - start-- 可选，查找的起始位置。
> - end-- 可选，查找的结束位置。

```python
>>> aList = [123, 'xyz', 'runoob', 'abc']
>>> aList.index( 'xyz' )
1
>>> aList.index( 'runoob', 1, 3 )
2
```

- list的其他常用方法

| 序号 | 方法                                                         |
| :--: | :----------------------------------------------------------- |
|  1   | list.append(obj) 在列表末尾添加新的对象                      |
|  2   | list.count(obj) 统计某个元素在列表中出现的次数               |
|  3   | list.extend(seq) 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表） |
|  4   | list.index(obj) 从列表中找出某个值第一个匹配项的索引位置     |
|  5   | list.insert(index, obj) 将对象插入列表                       |
|  6   | list.pop(index=-1\]) 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值 |
|  7   | list.remove(obj) 移除列表中某个值的第一个匹配项              |
|  8   | list.reverse() 反向列表中元素                                |
|  9   | list.sort(cmp=None, key=None, reverse=False) 对原列表进行排序 |

### 小结

通过实现特殊方法，自定义数据类型可以表现得跟内置类型一样，从而让我们写出更具表达力的代码——或者说，更具 Python 风格的代码。

# 2021.8.20

## RNN

RNN适用于序列数据，广泛应用于NLP领域。

用于图像分类领域，就要将处理序列数据的方法用于非序列数据。

<img src="note.assets/image-20210708105033837.png" alt="image-20210708105033837" style="zoom:50%;" />

<img src="../../../../准研一/2021-summer-vacation/Week 1/Note/cs231n_note.assets/image-20210708105051209.png" alt="image-20210708105051209" style="zoom: 50%;" />

- $W$​ 权重是共享的（$W_{hh}$​ $W_{xh}$​ $W_{hy}$​​ 均相同）。
- 输入和输出序列必须要是等长的。

**局限：长期依赖（Long-TermDependencies）问题**

![img](note.assets/quesbase6415569396815538507.png)

- RNN 会受到短时记忆的影响。如果一条序列足够长，那它们将很难将信息从较早的时间步传送到后面的时间步。

- 如果你正在尝试处理一段文本进行预测，RNN 可能从一开始就会遗漏重要信息，会面临梯度消失的问题。

- 梯度爆炸则是因为计算的难度越来越复杂导致。
- RNN是想把所有信息都记住，不管是有用还是没用的信息。

## LSTM

**长短期记忆（Long Short Term Memory，LSTM）网络**是一种特殊的RNN模型，其特殊的结构设计使得它可以避免长期依赖问题。即可以记住比较早之前出现的信息。

普通的RNN模型中，其重复神经网络模块的链式模型如下图所示，这个重复的模块只有一个非常简单的结构，一个单一的神经网络层（例如tanh层），这样就会导致信息的处理能力比较低。

<img src="note.assets/LSTM3-SimpleRNN.png" alt="img" style="zoom: 33%;" />

而LSTM在此基础上将这个结构进行了改进：

<img src="note.assets/LSTM3-chain.png" alt="A LSTM neural network." style="zoom: 33%;" />

LSTM的关键是细胞状态（直译：cell state），表示为 $C_t$ ，用来保存当前LSTM的状态信息并传递到下一时刻的LSTM中。当前的LSTM接收来自上一个时刻的细胞状态  $C_{t-1}$  ，并与当前LSTM接收的信号输入 $x_t$ 共同作用产生当前LSTM的细胞状态 $C_t$。细胞状态 $C_t$ 就代表着长期记忆。

<img src="note.assets/image-20210819170900925.png" alt="image-20210819170900925" style="zoom: 50%;" />

在LSTM中，采用专门设计的“门”来引入或者去除细胞状态  $C_t$ 中的信息。这里所采用的门包含一个sigmoid神经网络层和一个按位的乘法操作：<img src="note.assets/v2-6ba1445193a5731e297922efdde6559f_1440w.png" alt="img" style="zoom:36%;" /> 。0 表示“不允许任何量通过”，1 表示“允许所有量通过”。

LSTM主要包括三个不同的门结构：遗忘门、记忆门和输出门。

### 遗忘门

<img src="note.assets/LSTM3-focus-f.png" alt="img" style="zoom:36%;" />

遗忘门决定了细胞状态 $C_{t-1}$ 中的哪些信息将被遗忘。

### 记忆门

<img src="note.assets/LSTM3-focus-i.png" alt="img" style="zoom:36%;" />

记忆门的作用与遗忘门相反，它将决定新输入的信息 $x_t$ 和 $h_{t-1}$中哪些信息将被保留。

### 更新细胞状态

<img src="note.assets/LSTM3-focus-C.png" alt="img" style="zoom:36%;" />

利用遗忘门和记忆门，更新细胞状态 $C_t$。

### 输出门

<img src="note.assets/LSTM3-focus-o.png" alt="img" style="zoom:36%;" />

在 $t$ 时刻输入信号 $x_t$ 以后，计算对应的输出信号。

## Fluent python 第二章 序列构成的数组

### 推导式

推导式comprehensions（又称解析式），是Python的一种独有特性。推导式是可以从一个数据序列构建另一个新的数据序列的结构体，共有三种推导式。**原则上推导式尽量保持简短。**

#### 列表(list)推导式

```python
variable = [out_exp_res for out_exp in input_list if out_exp == 2]
```

- out_exp_res：列表生成元素表达式，可以是有返回值的函数。
- for out_exp in input_list：迭代input_list将out_exp传入out_exp_res表达式中。
- if out_exp == 2：根据条件过滤哪些值可以。

#### 字典(dict)推导式

```python
dictory = {key_expr: value_expr for value in collection if condition}
```

a.g.

```python
mcase_frequency = {v: k for k, v in mcase.items()}
```

#### 集合(set)推导式

```python
set = { expr for value in collection if condition }
```

a.g. 

```python
squared = {x**2 for x in [1, 1, 2]}
```

### 生成器表达式

- 生成器表达式（generator expression）则称为 genexps。

- 生成器表达式背后遵守了迭代器协议，可以逐个地产出元素，而不是先建立一个完整的列表，然后再把这个列表传递到某个构造函数里。前面那种方式显然能够节省内存。

- 生成器表达式的语法跟列表推导差不多，只不过把方括号换成圆括号而已。

a.g.

```python
>>> gen_exp = (x ** 2 for x in range(10) if x % 2 == 0) // 生成一个迭代器
>>> for x in gen_exp: // 用for循环进行迭代
...     print(x)
```

### 元组和记录

元组其实是对数据的记录：元组中的每个元素都存放了记录中一个字段的数据，外加这个字段的位置。

```python
>>> lax_coordinates = (33.9425, -118.408056) 
>>> city, year, pop, chg, area = ('Tokyo', 2003, 32450, 0.66, 8014) 
>>> traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'), 
... ('ESP', 'XDA205856')]
>>> for passport in sorted(traveler_ids): 
... print('%s/%s' % passport) ➎
...
BRA/CE342567
ESP/XDA205856
USA/31195855
```

如果在任何的表达式里我们在元组内对元素排序，元素所携带的信息就会丢失，这些信息是跟它们位置有关的。

### 元组拆包

- 元组拆包可以应用到任何**可迭代对象**上，唯一的硬性要求是，**被可迭代对象中的元素数量**必须要跟接受这些元素的元组的空档数一致。

```python
>>> import os
>>> _, filename = os.path.split('/home/luciano/.ssh/idrsa.pub')
>>> filename
'idrsa.pub'
```

- 在进行拆包的时候，我们不总是对元组里所有的数据都感兴趣，_ 占位符能帮助处理这种情况。

- 用 * 来处理剩下的元素。比如可能并不能确切地知道元组的长度，用*进行处理。

```python
>>> a, b, *rest = range(5) #前两个元素赋给a,b,剩余元素都赋给rest
>>> a, b, rest
(0, 1, [2, 3, 4])
```

### 切片

- 在切片和区间操作里不包含区间范围的最后一个元素是 Python 的风格，这个习惯符合Python、C 和其他语言里以 0 作为起始下标的传统。

- 当只有最后一个位置信息时，我们也可以快速看出切片和区间里有几个元素：`range(3)`和 `my_list[:3]` 都返回 3 个元素。

- 当起止位置信息都可见时，我们可以快速计算出切片和区间的长度，用后一个数减去第一个下标（stop - start）即可。

- 可以利用任意一个下标来把序列分割成不重叠的两部分，写成 `my_list[:x]` 和 `my_list[x:]` 就可以了

```python
>>> l = [10, 20, 30, 40, 50, 60] 
>>> l[:2] # 在下标2的地方分割
[10, 20] 
>>> l[2:] 
[30, 40, 50, 60] 
>>> l[:3] # 在下标3
```

一个众所周知的秘密是，我们还可以用 `s[a:b:c]` 的形式对 s 在 a 和 b 之间以 c 为间隔取值。c 的值还可以为负，负值意味着反向取值。

```python
>>> s = 'bicycle' 
>>> s[::3] #这种其实就是s[a:b:c]的变式
'bye' 
>>> s[::-1] 
'elcycib' 
>>> s[::-2] 
'eccb'
```

### 对序列使用+和*

```python
>>> l = [1, 2, 3]
>>> l + l
[1, 2, 3, 1, 2, 3]
>>> l * 5
[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
```

\+ 和 * 都遵循这个规律，不修改原有的操作对象，而是构建一个全新的序列。

