# 作业 2 报告

## 网络结构的设计及背后的理由

本作业中的神经网络结构借鉴了 ResNet 的思想，采用了带残差的卷积块来构建网络。网络的整体结构如下：

1. **卷积层（Convolutional Layers）**：我使用了三个卷积块，每个块包含两个卷积层，卷积核大小为 3x3，并且使用了 Batch Normalization 和 ReLU 激活函数。每一个卷积块都增加了直连通道，允许原始输入信息直接传到后面的层中，这样就可以同时提取浅层和深层的特征，提高网络的性能。
2. **池化层（Max Pooling Layers）**：在每个卷积块之后，我们使用了最大池化层来降低特征图的维度，提取出图像中的主要特征，同时减少参数数量，防止过拟合。
3. **全连接层（Fully Connected Layers）**：最后，我使用了两个全连接层，包含了两个 Dropout 层以及一个 ReLU 激活函数，在最后一层使用 Softmax 函数进行激活。
4. **损失函数和优化器（Loss Function & Optimizer）**：此网络使用 Cross Entropy 作为损失函数，同时加入 Lasso Regularization 防止过拟合，使用 Adam 作为优化器。

## 训练过程中所做的参数修改及调整

在训练过程中，我进行了一系列的参数修改和调整，以优化模型的性能和训练效果：

- **L1 正则化的添加**：为了防止模型过拟合，我在损失函数中添加了 L1 正则化项。通过控制权重参数，我们可以调整正则化的强度，最后选择权重参数为 1e-5。
- **批量大小**：在尝试了不同的批量大小后，最终选择了批量大小为 100。
- **学习率的调整**：经过测试，学习率为 0.001 时下降速度既不会太慢，振荡现象也可以得到控制。
- **损失函数的选择**：我最终在网络中采用了交叉熵损失函数。在模型训练的过程中，我也考虑过其他损失函数，如均方误差等，但最终选择了交叉熵损失函数，因其在多分类任务中的性能表现更好。

## 数据增强技术的使用

在数据预处理的时候，我们对图片进行了反转、随机裁剪、色彩抖动的操作

```python
from torchvision import transforms

# 数据增强的数据预处理方式
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])
```

在使用数据增强技术之后，由于每一个 epoch 的训练集和测试集的数据都不一样，训练过程中的过拟合现象会减少，并且最后训练出来的模型在测试集上的准确率也获得了一定的提高。

## 训练过程中遇到的挑战

在第一次尝试中，我使用了两个卷积层、两个池化层以及三个全连接处层，激活函数使用 ReLU 函数，但是出现了过拟合的问题。训练集的准确度在 100 个 epoch 之后可以达到 99% 以上，但是测试集上的准确率只能达到 60% 左右，因此需要进行正则化操作。

在全连接层中加上两层 Dropout 层后，训练集和测试集的正确率都在 70% 左右，因此我们需要对 CNN 进行进一步的调整。根据 VGG16 提供的灵感，我打算增加一些网络深度，而每一层的卷积核不需要太大。经过多次调整，网络给出的准确值依然只能达到 75% 左右。

于是我开始考虑一些新的结构，如 ResNet。通过搭建一个由三个 Residual Block 构成的神经网络，准确率就可以达到 79%，而最后的 1% 则需要通过正则化以及调整超参数得到。

## 最终的模型性能评估

在训练 100 个 epoch 之后，如果不使用数据增强技术，该网络在测试集上的正确率可以达到 80% 以上。

```
cifar10_cnn: without data augmentation
Epoch [100/100], Step [100/500], Loss: 0.3090, Accuracy: 95.00%
Epoch [100/100], Step [200/500], Loss: 0.4063, Accuracy: 93.00%
Epoch [100/100], Step [300/500], Loss: 0.4336, Accuracy: 95.00%
Epoch [100/100], Step [400/500], Loss: 0.3724, Accuracy: 91.00%
Epoch [100/100], Step [500/500], Loss: 0.3344, Accuracy: 96.00%
Test Accuracy of the model on the 10000 test images: 80.37 %
```

如果使用数据增强技术，则该网络在测试集上的正确率又能提高三个百分点。

```
Epoch [100/100], Step [100/500], Loss: 0.4682, Accuracy: 89.00%
Epoch [100/100], Step [200/500], Loss: 0.6134, Accuracy: 83.00%
Epoch [100/100], Step [300/500], Loss: 0.6883, Accuracy: 83.00%
Epoch [100/100], Step [400/500], Loss: 0.5271, Accuracy: 86.00%
Epoch [100/100], Step [500/500], Loss: 0.5989, Accuracy: 86.00%
Test Accuracy of the model on the 10000 test images: 83.99 %
```