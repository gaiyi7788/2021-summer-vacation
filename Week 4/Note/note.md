[TOC]

# 2021.7.31

## SSD

参考：[【SSD算法】史上最全代码解析-核心篇](https://zhuanlan.zhihu.com/p/79854543)     [SSD论文与代码详解](https://zhuanlan.zhihu.com/p/142630197)

针对Faster-RCNN存在的问题

> - 对小目标检测的效果很差（**较浅**层的特征图上，每个cell的**感受野**不是很大，所以适合检测**较小**的物体，而在**较深**的特征图上，每个cell的**感受野**就**比较大**了，适合检测**较大**的物体。）
> - 模型大，检测速度慢

该SSD网络由三部分组成，分别为：

> - 用于**图片特征提取**的网络：VGG base
>
> - 用于**引出多尺度特征图**的网络：Extra
>
> - 用于**框位置检测和分类**的网络：loc_layers和conf_layers

![image-20210731170445568](note.assets/image-20210731170445568.png)

### Backbone（VGG base）

根据SSD的论文描述，作者采用了vgg16的部分网络作为基础网络，在5层网络后，将FC6和FC7替换为conv6和conv7，分别为：1024x3x3、1024x1x1。

>  **值得注意：**
>
> 1. 与VGG-16一样，输入图像的大小恒为300×300；
> 2. conv4-1前面一层的maxpooling的ceil_mode=True,即向上取整，使得输出由37×37变为38x38；
> 3. Conv4-3网络是需要输出多尺度的网络层；
> 4. Conv5-3后面的一层maxpooling参数为(kernel_size=3, stride=1, padding=1)，不进行下采样。

![image-20210731171626469](note.assets/image-20210731171626469.png)

```python
def vgg(cfg, i, batch_norm=False):
    """根据cfg，生成类似VGG16的backbone"""
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # ceil模式就是会把不足square_size的边给保留下来，单独另算，或者也可以理解为在原来的数据上补充了值为-NAN的边
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
```

### Extra Layers

而为了后续的多尺度提取，在VGG Backbone后面添加了conv8、conv9、conv10、conv11。

<img src="note.assets/image-20210731172305629.png" alt="image-20210731172305629" style="zoom:60%;" /><img src="https://pic2.zhimg.com/80/v2-358084902079ece94e17f7d5cfc5a9ed_1440w.jpg" alt="img" style="zoom: 20%;" />

### Multi-box Layers

SSD选择了**6个特征图**作为**框位置检测和分类网络**的输入，其中**2个**来自**VGG base**,**4个**来自**Extra**。

![img](https://pic3.zhimg.com/80/v2-591e2a60166b51452fa076223c83b3aa_1440w.jpg)

**loc_layers** 和 **conf_layers** 是定义在函数 **multibox** 中的，用于**框位置检测和分类** 。在提取的**6个特征图**上的基础上引入**简单的一层3x3卷积层**进行位置信息和分类信息的提取，定义如下：

```python
def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    Args:
        vgg: 修改fc后的vgg网络
        extra_layers: 加在vgg后面的4层网络
        cfg: 网络参数，eg:[4, 6, 6, 6, 4, 4]，对应在不同的fp上cell为中心设定的待回归boxes的数量
        num_classes: 类别，VOC为 20+背景=21
    Return:
        vgg, extra_layers
        loc_layers: 多尺度分支的回归网络
        conf_layers: 多尺度分支的分类网络
    '''
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2] #在vgg上提取的两个特征图的位置
    # 在vgg backbone上引出两个分支，做位置回归和分类
    # 这里记录这两个分支的卷积网络
    for k, v in enumerate(vgg_source): #一共2个特征图
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                    cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                    cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 在extra layer上引出四个分支，做位置回归和分类
    # 这里记录这四个分支的卷积网络
    for k, v in enumerate(extra_layers[1::2], 2): #一共4个特征图
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)
```

Anchor是目标检测领域非常常见的一种技巧，在SSD原论文中，作者称之为**默认框**（default bounding boxes），在代码中，称之为**先验框**（prior)。在SSD中不同尺度的特征图上的cell，内置的**默认框/anchor**尺度是不同的，也就是**浅**的特征图负责检测**小**物体，所以较**浅**特征图的cell的**anchor尺寸较小**。

![image-20210731174559500](note.assets/image-20210731174559500.png)

SSD中共有**不同大小，不同位置**的anchor**8732个**

> 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732

先验框的设置，包括尺度（或者说大小）和长宽比两个方面。

<img src="note.assets/image-20210731181259323.png" alt="image-20210731181259323" style="zoom: 50%;" />

> 默认情况下，每个特征图会有一个 $a_r = 1$，且尺度为 $s_k$ 的先验框，除此之外，还会设置一个尺度为 $s_k^` = \sqrt{s_ks_{k+1}}$且 $a_r = 1$ 的先验框，这样每个特征图都设置了两个长宽比为1但大小不同的正方形先验框;

然后根据**面积和长宽比可得先验框的宽度和高度**:

<img src="note.assets/image-20210731181455577.png" alt="image-20210731181455577" style="zoom:67%;" />

然后将特征图上的点映射回原图上，生成anchor。

![image-20210731181808155](note.assets/image-20210731181808155.png)

可以看到，对于比较浅层的fp1，映射回原图中生成的实际的anchor的尺度小，而较深层的fp4对应的anchor的尺度比较大。印证了浅层特征图检测小目标，深层特征图检测大目标。

![image-20210731182640685](note.assets/image-20210731182640685.png)

每个位置有k个anchor（k为4 or 6），将每个anchor对应的feature map上的proposal经过3×3×((c+4)k)卷积，生成c个class score和4个offset。

> faster-rcnn是4×k×c个边界框回归参数，SDD是4×k个

### 样本提取

正样本：与任意GT的IOU大于0.5

负样本：**Hard negetive mining**，与所有GT都低于0.5为负样本，且每个负样本计算confidence loss，按照confidence loss由高到低排序，仅保留前面的负样本，使正样本：负样本=1: 3。（越容易被判断为object的负样本，confidence loss越大，即越难判断正确，保留这些样本做为负样本来训练）

### 损失函数

<img src="note.assets/image-20210731185526793.png" alt="image-20210731185526793" style="zoom:50%;" />

<img src="note.assets/image-20210731185557797.png" alt="image-20210731185557797" style="zoom: 50%;" />

# 2021.8.1-8.2

回老家了，要忙着走亲戚，这两天暂时没有办法学习。

