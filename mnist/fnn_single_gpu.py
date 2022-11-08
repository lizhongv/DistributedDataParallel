#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: Using FNN to train the data of MNIST, compared to ddp and dp....
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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # You cannot set a specific master process in this way, you can only default cuda:0 as the master process

# Initialize model, load_to GPU
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)


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
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
      
        # Forward pass
        outputs = model(images).cuda()  # batch_size * 10
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
torch.save(model.state_dict(), './data/model.ckpt')  

# load the model
model_new = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
model_new.load_state_dict(torch.load('./data/model.ckpt'))
# Load only model weights


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model_new(images)
        _, predicted = torch.max(outputs.data, 1)  # return（values, indices）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
