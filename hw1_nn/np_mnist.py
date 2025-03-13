# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (50000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    # ReLU function
    return np.maximum(0, x)

def relu_prime(x):
    # ReLU function's derivative
    return np.where(x>0, 1, 0)

def sigmoid(x):
    # Sigmoid function
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    # Sigmoid function's derivative
    return sigmoid(x) * (1. - sigmoid(x))

# 输出层激活函数
def f(x):
    # Softmax activate function
    # Add a parameter D to aviod overflow
    D = np.max(x)
    Z = np.exp(x - D).sum()
    return np.exp(x - D) / Z

def f_prime(x):
    # Softmax activate function's derivative
    D = np.max(x)
    Z = np.exp(x - D).sum()
    return np.exp(x - D) / Z * (1 - np.exp(x - D) / Z)

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    loss = - np.log(y_pred[np.argmax(y_true)])
    return loss

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    # Cross entropy loss
    # Since z and a are coupled together in the output layer (softmax activation function)
    # the derivative of the loss function with respect to z should involve a Jacobian matrix
    # dL/dz = dL/da * D(a)/D(z)
    # After some mathmetical derivation, we can get the derivative of the loss function
    # WITH RESPECT TO z
    loss_prime = y_pred - y_true
    return loss_prime


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    np.random.seed(seed=317)
    return np.random.normal(loc=0.0, scale=np.sqrt(1/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr # Learning rate

        # One hidden layer
        self.W1 = init_weights((input_size, hidden_size))
        self.b1 = init_weights((hidden_size,))
        self.WL = init_weights((hidden_size, output_size))
        self.bL = init_weights((output_size,))
        
    def forward(self, x):

        z1 = np.matmul(x, self.W1) + self.b1
        a1 = relu(z1)
        zL = np.matmul(a1, self.WL) + self.bL
        aL = f(zL) # a^{L}

        return z1, zL, a1, aL

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)
        self.grads_WL = np.zeros_like(self.WL)
        self.grads_bL = np.zeros_like(self.bL)

        batch_size = 0
        batch_loss = 0
        batch_acc  = 0
        for x, y in zip(x_batch, y_batch):
            # 前向传播
            z1, zL, a1, aL = self.forward(x)

            # 计算损失和准确率
            # cross entropy loss
            loss = loss_fn(y, aL)
            batch_size += 1
            batch_loss += loss
            batch_acc += (np.argmax(y) == np.argmax(aL))

            # 反向传播
            # 不同的激活函数，反向传播时的梯度计算方式不同
            # 1 : Sigmoid activation function
            # delta_L = (aL - y)
            # delta_1 = np.matmul(self.WL, delta_L) * a1 * (1 - a1)

            # 2 : ReLU activation function
            delta_L = (aL - y) # loss_fn_prime(y, aL)
            delta_1 = np.matmul(self.WL, delta_L) * relu_prime(z1)

            # Calculate gradients
            self.grads_WL += np.matmul(np.array([a1]).T, [delta_L])
            self.grads_bL += delta_L
            self.grads_W1 += np.matmul(np.array([x]).T, [delta_1])
            self.grads_b1 += delta_1

        # 梯度平均
        self.grads_WL /= batch_size
        self.grads_bL /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size

        batch_loss /= batch_size
        batch_acc /= batch_size

        train_losses.append(batch_loss)
        train_accuracies.append(batch_acc)

        # 更新权重
        self.WL -= self.lr * self.grads_WL
        self.bL -= self.lr * self.grads_bL
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1

if __name__ == '__main__':
    # 训练网络
    # The hidden size here does not matter too much here.
    # Smaller hidden size will accelerate the training process.
    net = Network(input_size=784, hidden_size=100, output_size=10, lr=0.3)
    for epoch in range(10):
        train_losses = []
        train_accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i+64]
            y_batch = y_train[i:i+64]
            net.step(x_batch, y_batch)
        
        print(f"Training Epoch:{epoch}")
        print(f"Training set: Average loss:{np.mean(train_losses)}, Accuracy:{np.mean(train_accuracies)}")
        # Validation
        val_acc = 0
        val_size = 0
        for x, y in zip(X_val, y_val):
            _, _, _, y_pred = net.forward(x)
            val_size += 1
            val_acc += (np.argmax(y) == np.argmax(y_pred))
        print(f"Validation set: Accuracy:{val_acc/val_size}")
        print()
            
    # Test
    test_acc = 0
    test_size = 0
    for x, y in zip(X_test, y_test):
        _, _, _, y_pred = net.forward(x)
        test_size += 1
        test_acc += (np.argmax(y) == np.argmax(y_pred))
    print("After training with 10 epochs: ")
    print(f"Testing set: Accuracy: {test_acc/test_size}")