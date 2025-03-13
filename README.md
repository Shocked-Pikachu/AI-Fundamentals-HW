# AI Fundamentals HW & Lab

Homework of Fundamentals of Artificial Intelligence (2024 Spring) @ School of EECS, Peking University

2024 春 人工智能基础课程作业/lab，北京大学信息技术科学学院

此项目记录我在 2024 春季选修的人工智能基础课程的部分作业（不包括学期末的大作业项目），虽然现在有许多知识已经忘却，但是本人还是在此课程中受益良多。尤其是在人工智能盛行的时代，我觉得这门课程即使作为一门通识课程也不足为过。

## 课程信息

- 开课时间：2024 年春季
- 开课院系：信息技术科学学院
- 授课教师：wly
- 课程简介：了解人工智能发展历史，掌握知识表达与逻辑推理、搜索、机器学习、深度学习、强化学习和博弈决策等基本算法，树立人工智能伦理和安全的意识，了解人工智能基本工具、芯片与平台，能够开展人工智能技术简单应用开发。

## 包含内容

- [实现训练 MLP 网络识别手写数字 MNIST 数据集](./hw1_nn)
- [实现卷积神经网络对 cifar10 数据进行分类](./hw2_cnn_cifar)
- [人工智能基础纸面作业](./hw3)
- [用 LSTM 对 IDMB 数据集进行分类](./hw4_imdb)
- [全局搜索/局部搜索/对抗搜索程序实例](./hw5-7)
- [房价预测模型](./hw8)
- [强化学习寻找冰湖上行走的最佳策略](./hw9)

## 其他说明

由于本人在 macOS 系统上运行代码，所以不能够使用 cuda，转而使用 mps

```python
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
```

若是在有 N 卡的 Windows 系统中运行，则可以直接使用 cuda

## 说在最后

这门课经过几年的改版和重组，到了现在的形态，可以说是差强人意。从知识的角度来说，这门课基本可以满足非智能专业对于人工智能的基础认识，但是对于部分常见模型（如扩散模型）的介绍有所欠缺。另外，从课程中的代码到实际生活中我们使用的模型还是有所差距。不过这也是能够理解的，毕竟对于面向低年级同学开设的这样一门企图包罗万象的介绍课程，也不能够要求在学完这门课程后能够独立开发一个有用的大模型。

从任务量的角度看，本课程相较于前几年已经大大减轻，将大部分的精力放在最主要的模型和算法上。作业的形式也是简单的填空和调试参数，对于一个三学分的课程是相对合适的，也对于我这样外院系的同学是比较友好的。

## License

Works in this repository is licensed under CC0, which is very similar to the Unlicense.

In other words, you may copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.