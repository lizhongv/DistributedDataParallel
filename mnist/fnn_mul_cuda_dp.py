#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: Using DP modle to train data of MNIST, to compate the training time, compared to single gpu and ddp....
"""

"""
单机多卡训练
torch.nn.DataParallel   DP方式

1. 设置可以被当前Python环境程序检测到的显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"  # 3为master卡， 这是实际显卡编号，而运算显卡编号为0,1,2
    或在命令行 CUDA_VISIBLE_DEVICES='1,2,3'
2. 设置master节点：  模型和向量必须先加载到该GPU中,默认为GPU0
    torch.cuda.set_device('cuda:1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # 这种写法无法指定master节点，默认cuda:0为master节点
3. 设置模型多卡并行  最好device_ids[0] = master节点，因为 默认下， output_device = device_ids[0]
    model = torch.nn.DataParallel(model, device_ids=[1,0,2], output_device=master节点).to(device)
    model = torch.nn.DataParallel(model, device_ids=[1,0,2], output_device=master节点).cuda('1')
4。加载模型到 master 节点
    x.to(device)
    x.cuda(1)
5. 保存模型
6. 加载模型
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import time

# Hyper-parameters
INPUT_SIZE = 784  # size of the picture 28*28
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHES = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# MNIST dataset    55000 train + 5000 valid  + 10000 test
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
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


os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,0,1"  # Actual GPU ID 3,2,1,0   Computing GPU ID 0,1,2,3（pytorch perceptual order）
torch.cuda.manual_seed_all(123)

# Device configuration
# torch.cuda.set_device('cuda:1')  # set specific master process
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # You cannot set a specific master process in this way, you can only default cuda:0 as the master process

# Initialize model, load_to GPU
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model = nn.DataParallel(model).cuda()

# model = nn.DataParallel(model, device_ids=[2,3], output_device=torch.cuda.current_device()).cuda() # device_id is computing GPU ID，setting device_ids[0] master process (cuda:2)
# model = nn.DataParallel(model, device_ids=[1,0,2]).to(device)
# torch.cuda.set_device('cuda:0')
print("model device:", next(model.parameters()).device)

# model parameters   batch default=-1
# summary(model,input_size=(INPUT_SIZE,), device='cuda')
# sum(x.numel() for x in model.parameters() if x.requires_grad==True)
# [INPUT(W)+1(b)]*hidden
# [hidden(W)+1(b)]*num_class


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # => nn.logSoftmax() + nn.LLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_time = time.time()

# Train the model
total_step = len(train_loader)
loss_list = []
for epoch in range(NUM_EPOCHES):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).cuda()
        # images = images.reshape(-1, 28 * 28).to(device)  # batch_size * input_size
        

        labels = labels.cuda()
        # labels = labels.to(device) # batch_size * 1
        
        # print(next(model.parameters()).device)
        # print(images.device)
        # print(labels.device)

        # Forward pass
        outputs = model(images).cuda()  # batch_size * 10
        # print(outputs.device)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHES, i + 1, total_step, loss))
        loss_list.append(loss.item())

print("total training time:", time.time()-start_time)

#  Visualize the loss
# def plt_loss():
#     plt.figure()
#     plt.plot(loss_list)
#     plt.xlabel('par step/100')
#     plt.ylabel('loss')
#     plt.title('LR=0.001')
#     plt.show()
#
# plt_loss()


# Save the model checkpoint 
# torch.save(model, './data/model.ckpt')  # save the whole model（structure+weight）
torch.save(model.state_dict(), './data/model.ckpt')  # Save the weights of the model

# load the model
model_new = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model_new = nn.DataParallel(model_new).cuda()

# model = torch.load('./data/model.ckpt')  # load the whole model
model_new.load_state_dict(torch.load('./data/model.ckpt'))  # Load only model weights
print('model_new device:', next(model_new.parameters()).device)


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).cuda()
        # print(images.device)
        labels = labels.cuda()
        # print(labels.device)

        outputs = model_new(images).cuda()
        # print(outputs.device)
        _, predicted = torch.max(outputs.data, 1)  # return（values, indices）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
