#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: test the function of lanuch.py 
"""

import os
import time
import torch.distributed as dist
 
print("before running dist.init_process_group()")
MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]
LOCAL_RANK = os.environ["LOCAL_RANK"]
RANK = os.environ["RANK"]
WORLD_SIZE = os.environ["WORLD_SIZE"]
 
print("MASTER_ADDR: {}\tMASTER_PORT: {}".format(MASTER_ADDR, MASTER_PORT))
print("LOCAL_RANK: {}\tRANK: {}\tWORLD_SIZE: {}".format(LOCAL_RANK, RANK, WORLD_SIZE))
 
dist.init_process_group('nccl',)
print("after running dist.init_process_group()")
time.sleep(20)  # Sleep for a while to avoid exceptions that occur when some processes end too quickly.
dist.destroy_process_group()



# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --MASTER_PORT=40895  testlaunch.py

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  testlaunch.py
