[TOC]

# 2021.8.12

## Fluent python 第一章

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

