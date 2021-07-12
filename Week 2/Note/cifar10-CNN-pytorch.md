[TOC]

# CNN网络——基于pytorch实现

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

`torchvision`主要包括一下几个包：

- **torchvision.datasets**: 几个常用视觉数据集，可以下载和加载，这里主要的高级用法就是可以看源码如何自己写自己的Dataset的子类
- **torchvision.models** : 例如 AlexNet, VGG, ResNet 和 Densenet 以及与训练好的参数。
- **torchvision.transforms** : 常用的图像操作，例如：随机切割，旋转，数据类型转换，图像到tensor ,numpy 数组到tensor , tensor 到 图像等。
- **torchvision.utils** : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。

## 数据集制作与数据预处理

```python
#通过Compose构造transform
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
) 

trainset = torchvision.datasets.CIFAR10(
    root='./data',train=True,download=True,transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',train=False,download=True,transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,batch_size=4,shuffle=True,num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset,batch_size=4,shuffle=False,num_workers=2
)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
```

### torchvision.transforms

```python
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
) 
```

`torchvision.transforms`是pytorch中的图像预处理包，包含了很多种对图像数据进行变换的函数。

`torchvision.transforms.Compose(transforms) `主要作用是将多个变换组合在一起，可以实现同时变换。即通过 Compose 方法构造 transform 可以同时进行多种不同变换。

`torchvision.transforms.ToTensor()`是将PILImage转变为torch.FloatTensor的数据形式；

`torchvision.transforms.Normalize(mean, std)`是用给定的均值和标准差分别对每个通道的数据进行正则化，每个通道都需要传入（如三通道图片传入(0.5,0.5,0.5)为三个均值）

### torchvision.datasets

使用 `torchvision.datasets` 可以轻易实现对这些数据集的训练集和测试集的下载，只需要使用 `torchvision.datasets` 再加上需要下载的数据集的名称就可以了。

```python
torchvision.datasets.CIFAR10(
    root='./data',train=True,download=True,transform=transform
)
```

其他常用的数据集如 `COCO`、`ImageNet`、`MNIST` 等都可以通过这个方法快速下载和载入。

- `root` 用于指定数据集在下载之后的存放路径，这里存放在根目录下的 `data` 文件夹中；
- `transform` 用于指定导入数据集时需要对数据进行哪种变换操作，见 1.1.2；
- `train` 用于指定在数据集下载完成后需要载入哪部分数据，
  - 如果设置为 `True`，则说明载入的是该数据集的训练集部分；
  - 如果设置为 `False`，则说明载入的是该数据集的测试集部分；

### torch.utils.data.Dataset

`torch.utils.data.Dataset`是代表自定义数据集方法的类，用户可以通过继承该类来自定义自己的数据集类，在继承时要求用户重载`__len__()`和`__getitem__()`这两个方法。

```python
# 预备：所有10000张图片（不区分类别）放在mnist_test文件夹内，并在当前目录下生成了一个mnist_test.txt的文件
# 格式： ./mnist_test/0.jpg label

from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset): #继承Dataset类
    def __init__(self, txt, transform=None, loader=default_loader):
        fh = open(txt, 'r') # 打开数据集txt，通过每行信息获取图片和标签
        imgs = []
        for line in fh: # 每行对应一个数据信息
            line = line.strip('\n') # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）
            line = line.rstrip() # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
            words = line.split() # 通过指定分隔符对字符串进行切片，默认为所有空字符，返回string list
            imgs.append((words[0],int(words[1]))) # 图片保存路径信息和标签共同构成元组
        self.imgs = imgs
        self.transform = transform # 设置transform
        self.loader = loader # 设置loader

    def __getitem__(self, index): # 通过index索引数据
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    dataset = MyDataset('data.txt')
```

- `__len__()`：**返回的是数据集的大小。**我们构建的数据集是一个对象，`__len__()`的目的就是获取对象的长度。
- `__getitem__()`：实现了能够通过索引的方法获取对象中的任意元素。可以在`__getitem__()`中实现数据预处理。

### torch.utils.data.random_split(*dataset*, *lengths*)

按照给定的长度将数据集划分成没有重叠的新数据集组合。

```python
train_size = int(0.8 * len(dataset)) # 划分结果要取整
test_size = len(dataset) - train_size # 根据取整结果，直接减得到
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
```

### torch.utils.data.DataLoader

```python
trainloader = torch.utils.data.DataLoader(
    trainset,batch_size=4,shuffle=True,num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset,batch_size=4,shuffle=False,num_workers=2
)
```

- `DataLoader`将`Dataset`对象或自定义数据类的对象封装成一个迭代器；
- 这个迭代器可以迭代输出`Dataset`的内容；
- 同时可以实现多进程、shuffle、不同采样策略，数据校对等等处理过程。

`__init__()`中的几个重要的输入：

- `dataset`：前面定义好的数据集（注意提前划分 `trainset` 和 `testset` ）
- `batch_size`：每个batch加载多少个样本(默认: 1)
- `shuffle`：随机打乱顺序，一般在训练数据中会采用。
- `num_workers`：这个参数必须大于等于0，0的话表示数据导入在主进程中进行，其他大于0的数表示通过多个进程来导入数据，可以加快数据导入速度。

## 定义网络模型

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

### torch.nn.Module

所有神经网络模块的基类，自己定义的模型应该继承这个类。

```python
class Net(nn.Module):
    def __init__(self):
        #super(Net,self).__init__() 
        super().__init__() #需要先通过nn.Moudle的__init__()函数初始化，再定义自己的网络
```

- super()函数是用于调用父类(超类)的一个方法。

- Python 3 可以使用直接使用 **super().xxx** 代替 **super(Class, self).xxx** :

### torch.nn.Conv2d(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, . . . )

- Applies a 2D convolution over an input signal composed of several input planes.

```python
self.conv1 = nn.Conv2d(3,6,5)
```

- 对上例可以理解成对一个三通道的输入图像用6个5*5的卷积核进行操作，因此输出是6通道。

- 其余属性用到再查手册。

### torch.nn.MaxPool2d(*kernel_size*, *stride=None*, *padding=0*, . . .)

```python
self.pool = nn.MaxPool2d(2,2) # kernel_size=2 , stride=2
```

- Applies a 2D convolution over an input signal composed of several input planes.

- 采用的是Maxpooling。

- **stride** – the stride of the window. Default value is `kernel_size`. 非常合理，只有如果kernel_size取2，只有当stride也是2是pooling才能实现下2采样。

