# Faster-rcnn

## 图像预处理tranform

![image-20210724000027243](faster-rcnn最终总结.assets/image-20210724000027243.png)

### normalize

$$
image = \frac{image-mean}{std}
$$



归一化处理，改变了像素值，但是图片尺寸未改变。

### resize

将图片尺寸调整到设定的上下限内。

- 设定两个参数，允许的最大尺寸*self_max_size*和允许的最小尺寸*self_min_size*

- 得到待处理图片较大的边长*max_size*和较小的边长*min_size*

- 缩放因子*scale_factor* = *self_min_size* / min_size

- ```python
  if max_size * scale_factor > self_max_size:
  	scale_factor = self_max_size / max_size  
      # 将缩放比例设为指定最大边长和图片最大边长之比
  ```

- 根据scale_factor对原图进行双线性插值，得到resize后的图片

- bbox 也要根据scale_factor进行缩放

### batch_images

将处理完的图片打包成一个batch

- 先得到 batch 中所有图片最大的一个尺寸（宽的最大值，高的最大值），设为batch_shape
- 设定stride=32（一般是用于加速计算）
- 将每个图片的尺寸向上取整到stride的整数倍
- 根据batch_shape生成全0的矩阵，将待处理图片复制到矩阵左上角

<img src="faster-rcnn最终总结.assets/image-20210724011157658.png" alt="image-20210724011157658" style="zoom:50%;" />

- 这样保证了一个batch中的所有图片的尺寸相同

至此图片预处理完成，可以传入backbone网络。

## 特征提取网络 backbone

