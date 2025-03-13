# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


# 定义超参数
batch_size = 100
learning_rate = 0.001
num_epochs = 100

# 定义数据预处理方式
# 普通的数据预处理方式
# transforms.Compose() creates a series of transformations (in the list)
# to be applied to the input data.
# transforms.ToTensor() converts the input data to a tensor.
# Here, we only need the original data.
# transform = transforms.Compose([
#     transforms.ToTensor(),])

# 数据增强的数据预处理方式
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])



# 定义数据集
# 数据集包含60000张32*32的彩色图片，共10个类别
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    Define the convolutional neural network
    2 convolutional layers &
    3 fully connected layers
    '''
    def __init__(self):
        super(Net, self).__init__()
        # # Define the convolutional layers and pooling layers
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(6, 10, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        self.block1 = self.basic_block(3, 16)
        self.downsample1 = nn.Conv2d(3, 16, kernel_size=1)  # 1x1 conv to match the channels
        self.block2 = self.basic_block(16, 32)
        self.downsample2 = nn.Conv2d(16, 32, kernel_size=1)  # 1x1 conv to match the channels
        self.block3 = self.basic_block(32, 64)
        self.downsample3 = nn.Conv2d(32, 64, kernel_size=1)  # 1x1 conv to match the channels

        # Define the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Add a dropout layer
            nn.Linear(128, 10),
        )
    
    def basic_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # Batch normalization is added to stablize the training
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # First residual block
        residual = self.downsample1(x)
        x = self.block1(x)
        x = residual + x
        x = F.relu(x)

        x = F.max_pool2d(x, 2, 2)

        # Second residual block
        residual = self.downsample2(x)
        x = self.block2(x)
        x = residual + x
        x = F.relu(x)

        x = F.max_pool2d(x, 2, 2)

        # Third residual block
        residual = self.downsample3(x)
        x = self.block3(x)
        x = residual + x
        x = F.relu(x)

        x = F.max_pool2d(x, 2, 2)

        # Flatten the tensor
        x = x.view(-1, 64 * 4 * 4)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


# 实例化模型
model = Net()

use_mlu = False
try:
    # Check if MLU is available
    # However, on my device, mlu is not available
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    # if torch.cuda.is_available():
    '''
    On macOS, we use MPS instead of CUDA
    '''
    if torch.backends.mps.is_available():
        device = torch.device('mps:0')
    else:
        device = torch.device('cpu')

model = model.to(device)

# 定义损失函数和优化器
# criterion = None
criterion = nn.CrossEntropyLoss()
# optimizer = None
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Add the L1 regularization
        l1_lambda = 1e-5
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))