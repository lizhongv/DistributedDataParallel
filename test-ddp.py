#!/usr/env/bin python
# -*- coding:utf-8 -*-
"""
@Author: LiZhong
@Email: zl2272001@gmail.com
@Time: 2022/11/19
@target: https://blog.csdn.net/searobbers_duck/article/details/115299691
"""

import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision

from torch.utils.data import Dataset, DataLoader

import numpy as np


class InnerDS(Dataset):
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        np_img = np.random.rand(3, 224, 224)
        image = torch.from_numpy(np_img).float()
        label = np.random.randint(0, 9)
        return image, label, key


local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

# os.environ.keys()
print('master_addr:', os.environ['MASTER_ADDR'])
print('master_port', os.environ['MASTER_PORT'])
print('rank:', os.environ['RANK'])
print('local_rank:', os.environ['LOCAL_RANK'])
print('world_size:', os.environ['WORLD_SIZE'])

dist.init_process_group('nccl', world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)

# case 1
# ds = InnerDS(8)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, drop_last=True)

# case 2
# ds = InnerDS(8)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True)

# case 3
# ds = InnerDS(8)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True)

# case 4
# ds = InnerDS(6)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True)


# case 5
# ds = InnerDS(5)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=False)

# case 6
ds = InnerDS(10)
sampler = torch.utils.data.distributed.DistributedSampler(ds)
dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=False)

# case 7
# ds = InnerDS(10)
# sampler = torch.utils.data.distributed.DistributedSampler(ds)
# dataloader = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=True)


for epoch in range(2):
    # case 3+
    sampler.set_epoch(epoch) #
    for index, (_, labels, keys) in enumerate(dataloader):
        print(f'{index}/{epoch}:\t', keys.cuda())
        dist.barrier()

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr='127.0.0.1' --master_port='29500' 123.py
