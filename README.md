## local_rank
> 单进程会设置进程号为 `-1`，总之当 `local_rank=-1/0` 可以认为主进程
> 多进程处理数据时让其他进程全部停下来

```python
from contextlib import contextmanager

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something 确保分布式训练中主进程完毕
    if local_rank not in [-1, 0]:  # 当前进程是否为主进程
        # 不是就让pytorch对它进行阻塞，也就是暂停运行
        torch.distributed.barrier(device_ids=[local_rank])
    yield  # 如果他是主进程，他就暂时跳出函数执行其他任务
    if local_rank == 0: # 此时主进程完成了其他任务进入了第二个if函数
        # 然后它也进入barrier（）函数里面
         torch.distributed.barrier(device_ids=[0])
# 当pytorch发现所有的进程都进入了barrier（），就会打开所有的barrier，所有的进程都可以继续进行。
```
[关于torch.distributed.barrier()的作用](https://blog.csdn.net/REstrat/article/details/126881877?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-126881877-blog-109820870.235%5Ev28%5Epc_relevant_t0_download&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-126881877-blog-109820870.235%5Ev28%5Epc_relevant_t0_download&utm_relevant_index=2)


## DistributedDataParallel
this a simple dataparallel python application


## Pytorch中将模型和张量加载到GPU的常用方法有两种。

**方法一**

```python
# 如果GPU可用，将模型和张量加载到GPU上
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    y = y.cuda()
```
**方法二**
```python
# 分配到的GPU或CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型加到GPU
model = model.to(device)
# 将张量加到GPU
x = x.to(device)
y = y.to(device)
```

<div align=left><img src="https://user-images.githubusercontent.com/49021250/230571742-ff2e2223-bb4f-4e1e-8100-942b8ee6b1d5.png" width="50%" height="50%"></div>


https://blog.csdn.net/Pl_Sun/article/details/106978171
## 注意model 和model.module区别
