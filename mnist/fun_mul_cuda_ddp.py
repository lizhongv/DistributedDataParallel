#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: Using DP modle to train data of MNIST, to compate the training time, compared to single gpu and ddp....
"""

"""
单机多卡训练
nn.parellel.DistributedDataParallel

1. 从外面得到local_rank参数   在调用DDP的时候，会自动给出这个参数
2. 根据local_rank 来设定当前使用那块GPU   torch.cuda.set_device(args.local_rank)
3。初始化进程组   dist.init_process_group(backend='nccl', init_method='env://')
4. 数据集分享  sampler_train = DistributedSampler(train_dataset)
5. 模型并行   DistributedDataParallel(model)
6. 启动工具  torch.distributed.launch  在每个主句上执行一次script，开始训练
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torch.nn.parallel import DistributedDataParallel
import time


# 4. 命令参数解析
parser = argparse.ArgumentParser(description="training")
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for dist')
args = parser.parse_args()

# local_rank 和 world_size   单机时进程就是GPU，二者相等
local_rank = args.local_rank  # 本地的进程号/GPU编号
world_size = torch.cuda.device_count()  # 全局进程数
# world_size =  dist.get_world_size()

# =====================================================
# 5. 初始化进程组
torch.cuda.set_device(local_rank)
# print(dist.is_nccl_available())
dist.init_process_group(backend='nccl', init_method='env://')
# backend：指后节点或进程间在后端通信方式（常用三种mpi, nccl, gloo）。
# init_method：指定如何初始化进程组的URL。
# word_size: 参与job的进程数。 与可用GPU数量一致
# rank: 当前进程的等级。    rank=0为master节点
# local_rank: 本地进程或GPU编号， 每个主机/节点有多个进程
# rank=3，local_rank=0：表示第3个节点内的第1块GPU


# print('local_rank:', args.local_rank)
# print('current process rank:', dist.get_rank())
# # print('backend:', dist.get_backend())
# print('world_size:', dist.get_world_size())
# print('current_device:', torch.cuda.current_device())
# # print('MASTER_ADDR:', os.environ['MASTER_ADDR'])  # 因为单机 localhost
# # print('MASTER_PORT:', os.environ['MASTER_PORT'])
# # print("WORLD_SIZE:", os.environ['WORLD_SIZE'])
# print('RANK', os.environ['RANK'])  # 绝对进程GPU编号（单主机时等价于local_rank）

#  Device configuration  根据local_rank配置当前进程使用的GPU
device = torch.device('cuda', local_rank)  # 本地的进程/GPU
# torch.device('cuda:0)
# torch.device('cuda', 0)

# =============================================================
# 1. Hyper-parameters
INPUT_SIZE = 784  # 28x28（一张图片）
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHES = 50
BATCH_SIZE = 128  # 一个batch更新一次参数
LEARNING_RATE = 0.001

# ==========================================================================
# 2. MNIST dataset    { 55000张train + 5000张valid  + 10000张test}
train_dataset = MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

test_dataset = MNIST(root='./data',
                     train=False,
                     transform=transforms.ToTensor(),
                     download=True)

sampler_train = DistributedSampler(train_dataset)
sampler_test = DistributedSampler(test_dataset)

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=sampler_train,
                          drop_last=False)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         sampler=sampler_test,
                         drop_last=False)


# 3. Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # [batch, d_model]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # [batch, num_class]
        return out

# 6. Initialize model, load_to GPU
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
print(next(model.parameters()).device)


# 7. Loss and optimizer
criterion = nn.CrossEntropyLoss()  # => nn.logSoftmax() + nn.LLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


start_time = time.time()

# 8. Train the model
total_step = len(train_loader)
loss_list = []
for epoch in range(NUM_EPOCHES):
    sampler_train.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)  # batch_size * input_size
        # print('images:', images.device, images.shape) 
        labels = labels.to(device)  # batch_size * 1
        # print('labels:', labels.device, labels.shape)

        # Forward pass
        outputs = model(images).to(device)  # batch_size * 10
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播, 计算梯度值
        optimizer.step()  # 参数更新


        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHES, i + 1, total_step, loss))
        loss_list.append(loss.item())

print("total training time:", time.time()-start_time)
#
# # 可视化 loss
# def plt_loss():
#     plt.figure()
#     plt.plot(loss_list)
#     plt.xlabel('par step/100')
#     plt.ylabel('loss')
#     plt.title('LR=0.001')
#     plt.show()
#
#
# plt_loss()
#
# # Save the model checkpoint
# torch.save(model.state_dict(), './data/model1.ckpt')
#
# # load the model
# model_new = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
# model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
# model_new.load_state_dict(torch.load('./data/model1.ckpt'))
#
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28 * 28).to(device)
#         labels = labels.to(device)
#
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)  # return（values, indices）
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()  # 预测正确的个数
#
#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
#
# # 一台主机就是一个节点  一个进程就是一个GPU
# # torch.distributed.launch
# # 为当前主机/节点创建 nproc_pro_node 个进程，每个进程独立执行训练脚本，
# #  local_rank：表示当前进程在当前主机/节点上的编号
# #  rank: 表示所有节点上的绝对进程编号   单机上：rank=local_rank
#
#
# ## nproc_per_node：为当前主机/节点创建的进程数，一般设定为当前主机的GPU数
# ## nnodes：当前job包含多少个节点
# ## node_rank：指定当前主机/节点的序号/优先级
# ## master_addr 和 master_port: 分别指定 master 节点的 ip 和 port
#
# # CUDA_VISIBLE_DEVICES=1,3,2,0 python -m torch.distributed.launch --nproc_per_node=4 fnn_mul_cuda_ddp.py
