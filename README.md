# DistributedDataParallel
this a simple dataparallel python application


# Pytorch中将模型和张量加载到GPU的常用方法有两种。

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
