#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: Using single GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 40
data_size = 100

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dummy Dataset
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
# dataloader
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

# demo model - linear operation
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print(f"In Model:", "input",input.shape, input.device,
              " output:", output.size(), output.device)
        return output
   
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!", f"batch_size = {batch_size}")
model.to(device)

# training data
# print the size of input tensors and output tensors 
for i, data in enumerate(rand_loader):
    input = data.to(device)
    output = model(input)
    print(f"[{i}]:", "Outside: input size", input.size(),
          " output_size", output.size())
