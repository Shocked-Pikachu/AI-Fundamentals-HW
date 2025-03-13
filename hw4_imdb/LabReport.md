# HW4-IMDB

## 作业 1 LSTM RNN GRU 对比试验

### `torch.nn.RNN` 输入和输出结构

#### 输入：input, h_0

- **input** - 对非批量数据，其形状为 *(seq, input_size)*，其中 *seq* 是序列的长度（单词数量），*input_size* 是输入数据的特征大小（每个单词的嵌入长度）。对批量数据，其形状为 *(seq, batch, input_size)* 或者 *(batch, seq, input_size)*，取决于 `batch_first` 这一参数的设置
- **h_0** - 输出隐藏向量，对非批量数据，其形状为 *(num_layers, hidden_size)*，对批量数据，其形状为 *(num_layers, batch, hidden_size)*。如果不提供这个向量的话默认设置为 0

#### 输出：output, h_n

- **output** - 对非批量数据，其形状为 *(seq, hidden_size)*。对批量数据，其形状为 *(seq, batch, hidden_size)* 或者 *(batch, seq, hidden_size)*
- **h_n** - 最后一层隐藏层的隐藏向量，对非批量数据，其形状为 *(num_layers, hidden_size)*，对批量数据，其形状为 *(num_layers, batch, hidden_size)*

### `torch.nn.RNN` 初始化参数

```python
class RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None):
```

- `input_size` - 输入数据 *x* 的特征大小，即每个时间输入的向量的维度

- `hidden_size` - 隐藏层 *h* 的特征大小，其表明了模型的记忆能力

- `num_layers` - 循环层的个数。例如设置 `num_layers=2` 就是将两个 RNN 合并成一个大的 RNN 网络，第二个 RNN 会将第一个 RNN 的输出结果作为输入并且计算最终结果。默认值为 1

- `nonlinearity` - 使用的非线性激活函数。默认值为 `'tanh'`，其意思是对每个输入序列，RNN 会计算如下函数
  $$
  h_t=\tanh(x_tW_{ih}^T+b_{ih}+h_{t-1}W_{hh}^T+b_{hh})\nonumber
  $$
  其中 $h_t$ 是在时刻 $t$ 的隐藏向量，$x_t$ 是在时刻 $t$ 的输入，$h_{t-1}$ 是在时刻 $(t-1)$​ 的隐藏向量

- `bias` - 如果是 `False` 则不会使用 bias 项 $b_{ih}$ 和 $b_{hh}$​。默认值为 `True`

- `batch_first` - 如果是 `True` 则输入和输出的向量都会变为 *(batch, seq, feature)* 而不是 *(seq, batch, feature)* 的形状。默认值为 `False`

- `drop_out` - 如果设置了非零数，那么就会在 RNN 中添加一个 Dropout 层，dropout 的概率为 `drop_out`。默认值为 0

- `bidirectional` - 如果是 `True` 则网络会变成双向 RNN。默认值为 `False`

### 不同 Layer 的运行结果

#### RNN 的运行结果

```log
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (rnn): RNN(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.595227    Acc: 0.671226
Test set: Average loss: 0.4970, Accuracy: 0.7634
Train Epoch: 2 Loss: 0.423615    Acc: 0.809704
Test set: Average loss: 0.4179, Accuracy: 0.8127
Train Epoch: 3 Loss: 0.328654    Acc: 0.865765
Test set: Average loss: 0.3899, Accuracy: 0.8388
Train Epoch: 4 Loss: 0.264845    Acc: 0.895118
Test set: Average loss: 0.3492, Accuracy: 0.8491
Train Epoch: 5 Loss: 0.214282    Acc: 0.921625
Test set: Average loss: 0.4127, Accuracy: 0.8418
```

#### LSTM 的运行结果

```log
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.639746    Acc: 0.624251
Test set: Average loss: 0.5870, Accuracy: 0.6873
Train Epoch: 2 Loss: 0.501648    Acc: 0.754942
Test set: Average loss: 0.4641, Accuracy: 0.7733
Train Epoch: 3 Loss: 0.425447    Acc: 0.803964
Test set: Average loss: 0.4445, Accuracy: 0.7876
Train Epoch: 4 Loss: 0.383797    Acc: 0.828774
Test set: Average loss: 0.3991, Accuracy: 0.8133
Train Epoch: 5 Loss: 0.347000    Acc: 0.847943
Test set: Average loss: 0.4060, Accuracy: 0.8159
```

#### GRU 的运行结果

```log
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (gru): GRU(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.570131    Acc: 0.685903
Test set: Average loss: 0.4451, Accuracy: 0.7925
Train Epoch: 2 Loss: 0.371094    Acc: 0.837710
Test set: Average loss: 0.3627, Accuracy: 0.8370
Train Epoch: 3 Loss: 0.290890    Acc: 0.879493
Test set: Average loss: 0.3399, Accuracy: 0.8513
Train Epoch: 4 Loss: 0.233760    Acc: 0.909345
Test set: Average loss: 0.3525, Accuracy: 0.8499
Train Epoch: 5 Loss: 0.184220    Acc: 0.930960
Test set: Average loss: 0.3520, Accuracy: 0.8590
```

## 作业 2 手写 LSTM 实验

本 lab 参考 Pytorch 官方文档 [LSTM — PyTorch 2.2 documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) 进行编写

对于每一个输入的序列，LSTM 网络计算如下的函数
$$
f_t={\rm sigmoid}(W_{if}x_t+W_{hf}h_{t-1}+b_f)\\
i_t={\rm sigmoid}(W_{ii}x_t+W_{hi}h_{t-1}+b_i)\\
g_t=\tanh(W_{ig}x_t+W_{hg}h_{t-1}+b_g)\\
o_t={\rm sigmoid}(W_{io}x_t+W_{ho}h_{t-1}+b_o)\\
c_t=f_t\odot c_{t-1}+i_t\odot g_t\\
h_t=o_t\odot\tanh(c_t)\nonumber
$$
网络对一个序列中所有的 hidden vectors 求平均得到输出，之后用两个线性层（其中有 ReLU 激活层和 Dropout 层）对输出进行分类

> :warning:注意：由于本代码是在 macOS 上跑通的，如果想在 windows 系统上跑通的话需要将 34 行改为 `device = torch.device('cuda:0')`

### 手写 LSTM 参考结果

```log
Net(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM()
  (fc1): Linear(in_features=64, out_features=32, bias=True)
  (act): ReLU()
  (dropout): Dropout(p=0.4, inplace=False)
  (fc2): Linear(in_features=32, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.669380    Acc: 0.578275
Test set: Average loss: 0.5621, Accuracy: 0.7429
Train Epoch: 2 Loss: 0.541926    Acc: 0.739816
Test set: Average loss: 0.4626, Accuracy: 0.8070
Train Epoch: 3 Loss: 0.456821    Acc: 0.802067
Test set: Average loss: 0.4259, Accuracy: 0.8250
Train Epoch: 4 Loss: 0.413881    Acc: 0.826877
Test set: Average loss: 0.4435, Accuracy: 0.8252
Train Epoch: 5 Loss: 0.359966    Acc: 0.841953
Test set: Average loss: 0.3946, Accuracy: 0.8408
```

### 调整网络结构以观察影响

#### 调整隐藏层维度

将 `hidden_size` 调整为 32

```log
Net(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM()
  (fc1): Linear(in_features=32, out_features=32, bias=True)
  (act): ReLU()
  (dropout): Dropout(p=0.4, inplace=False)
  (fc2): Linear(in_features=32, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.508449    Acc: 0.746306
Test set: Average loss: 0.4107, Accuracy: 0.8254
Train Epoch: 2 Loss: 0.323906    Acc: 0.868710
Test set: Average loss: 0.4125, Accuracy: 0.8178
Train Epoch: 3 Loss: 0.264356    Acc: 0.899062
Test set: Average loss: 0.3767, Accuracy: 0.8370
Train Epoch: 4 Loss: 0.226558    Acc: 0.915335
Test set: Average loss: 0.4448, Accuracy: 0.8325
Train Epoch: 5 Loss: 0.202243    Acc: 0.924371
Test set: Average loss: 0.4407, Accuracy: 0.8372
```

`hidden_size` 减少对准确率影响不大，但有意思的是当 `hidden_size` 增加到 128 的时候，准确率反而会下降。

#### 调整损失函数

将 loss function 调整为 NLLLoss (the negative log likelihood loss)

```log
Net(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM()
  (fc1): Linear(in_features=64, out_features=32, bias=True)
  (act): ReLU()
  (dropout): Dropout(p=0.4, inplace=False)
  (fc2): Linear(in_features=32, out_features=2, bias=True)
)
Train Epoch: 1 Loss: -1675127892215.033203       Acc: 0.500799
Test set: Average loss: -15.1988, Accuracy: 0.4939
Train Epoch: 2 Loss: -22.998989          Acc: 0.501548
Test set: Average loss: -30.8482, Accuracy: 0.4939
Train Epoch: 3 Loss: -38.649096          Acc: 0.501498
Test set: Average loss: -46.4975, Accuracy: 0.4939
Train Epoch: 4 Loss: -54.299359          Acc: 0.501697
Test set: Average loss: -62.1469, Accuracy: 0.4939
Train Epoch: 5 Loss: -69.949388          Acc: 0.501398
Test set: Average loss: -77.7965, Accuracy: 0.4939
```

损失函数对训练的结果影响比较大。

#### 调整训练流程

将 `epoch` 调整为 10，将 `batch_size` 调整为 128

```log
Net(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM()
  (fc1): Linear(in_features=64, out_features=32, bias=True)
  (act): ReLU()
  (dropout): Dropout(p=0.4, inplace=False)
  (fc2): Linear(in_features=32, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.680949    Acc: 0.559365
Test set: Average loss: 0.6485, Accuracy: 0.6436
Train Epoch: 2 Loss: 0.588898    Acc: 0.695014
Test set: Average loss: 0.5211, Accuracy: 0.7443
Train Epoch: 3 Loss: 0.447769    Acc: 0.797721
Test set: Average loss: 0.4456, Accuracy: 0.7932
Train Epoch: 4 Loss: 0.339512    Acc: 0.855643
Test set: Average loss: 0.4491, Accuracy: 0.8012
Train Epoch: 5 Loss: 0.271869    Acc: 0.886346
Test set: Average loss: 0.4542, Accuracy: 0.8088
Train Epoch: 6 Loss: 0.236991    Acc: 0.903513
Test set: Average loss: 0.4651, Accuracy: 0.8145
Train Epoch: 7 Loss: 0.198269    Acc: 0.918840
Test set: Average loss: 0.5211, Accuracy: 0.8166
Train Epoch: 8 Loss: 0.174067    Acc: 0.927000
Test set: Average loss: 0.6056, Accuracy: 0.8176
Train Epoch: 9 Loss: 0.154161    Acc: 0.932126
Test set: Average loss: 0.6601, Accuracy: 0.8063
Train Epoch: 10 Loss: 0.145143   Acc: 0.931479
Test set: Average loss: 0.5974, Accuracy: 0.8160
```

`epoch` 和 `batch_size` 都很难显著提升模型在测试集上的表现，随着 `epoch` 数目的增加，训练集的准确率越来越高，说明出现了过拟合现象，如果加入一些正则化会期望模型的效果变得更好。
