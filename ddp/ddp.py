#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: Using DDP model to train data, using torch.distributed.launch to start multi-process...
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import argparse

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    # The most important step of the procedure
    dist.init_process_group(backend="nccl")
    
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()
    
def demo_basic(local_world_size, local_rank):

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    ranks = list(range(local_rank * n, (local_rank + 1) * n))
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {ranks}"
    )

    model = ToyModel().cuda(ranks[0])
    # The most important step of the procedure
    ddp_model = DDP(model, device_ids=ranks)   # default: outputs_ids=device_ids[0]=ranks[0]

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(ranks[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='local rank ID')  
    parser.add_argument("--local_world_size", type=int, default=1, help="the number of local process")
    args = parser.parse_args()

    # --local_rank, --local_world_size 
    spmd_main(args.local_world_size, args.local_rank)



# python -m  torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4 ddp.py --local_world_size=4 

# print launch.py location
# (py3.6_torch1.7.1) lizhong@amax:/data0/lizhong/DDP$ python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))"
# /data0/lizhong/anaconda3/envs/py3.6_torch1.7.1/lib/python3.6/site-packages/torch/distributed/launch.py

# python /data0/lizhong/anaconda3/envs/py3.6_torch1.7.1/lib/python3.6/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=4 ddp.py --local_world_size=4
