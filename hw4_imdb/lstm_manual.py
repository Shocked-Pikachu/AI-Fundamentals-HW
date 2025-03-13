# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys

from utils import load_imdb_dataset, Accuracy
from tqdm import tqdm

use_mlu = False
# try:
#     import torch_mlu
#     import torch_mlu.core.mlu_model as ct
#     global ct
#     use_mlu = torch.mlu.is_available()
# except:
#     use_mlu = False

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

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1


class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


# 你需要实现的手写LSTM内容，包括LSTM类所属的__init__函数和forward函数
class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define weights and biases

        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.W_ig = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

    # LSTM层
    def forward(self, x, hidden):
        """
        x: input sequence, shape: (seq_len, batch_size, input_size)
        hidden: initial value of the hidden state, shape: (batch_size, hidden_size)
        """
        seq_len = x.shape[0]
        h, c = hidden
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            # x_t shape: (batch_size, input_size)
            # Forget gate
            f_t = torch.sigmoid(torch.mm(x_t, self.W_if) + torch.mm(h, self.W_hf) + self.b_f)
            # Input gate
            i_t = torch.sigmoid(torch.mm(x_t, self.W_ii) + torch.mm(h, self.W_hi) + self.b_i)
            # Output gate
            g_t = torch.tanh(torch.mm(x_t, self.W_ig) + torch.mm(h, self.W_hg) + self.b_g)
            o_t = torch.sigmoid(torch.mm(x_t, self.W_io) + torch.mm(h, self.W_ho) + self.b_o)
            # Update cell state
            c = f_t * c + i_t * g_t
            h = o_t * torch.tanh(c)
            # After unsqueezing, h shape: (1, batch_size, hidden_size)
            outputs.append(h.unsqueeze(dim=0))
        
        # Outputs shape (after concatenating): (seq_len, batch_size, hidden_size)
        outputs = torch.cat(outputs, dim=0)

        return outputs

# 你需要实现网络推理和训练内容，仅需要完善forward函数
class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''

    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        '''
        x: 输入, shape: (batch_size, seq_len)
        '''

        # 词嵌入
        x = self.embedding(x)
        # LSTM layer
        batch_size = x.shape[0]
        prev_h = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        prev_c = torch.zeros(batch_size, self.lstm.hidden_size).to(device)
        x = x.permute(1, 0, 2) # transpose x to (seq_len, batch_size, input_size)
        x = self.lstm(x, [prev_h, prev_c])
        # Full connect layers
        # Take the mean of the hidden states of all time steps as the output
        x = torch.mean(x, dim=0)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


n_epoch = 10
batch_size = 128
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    # loss_func = torch.nn.NLLLoss(reduction="mean")
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    # loss_func = torch.nn.NLLLoss(reduction="mean").
    test_loss = 0
    test_acc = 0
    n_iter = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            metric.update(output, target)
            test_acc += metric.result()
            metric.reset()
            n_iter += 1
    test_loss /= n_iter
    test_acc /= n_iter
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


optimizer = torch.optim.Adam(net.parameters(), lr=5e-2, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
