```python
batch_size 要大于GPU数量
device_ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 或者 'cuda:0' 指的主卡
model.to(device) # 放在主卡
data.to(device) # 放在主卡
损失函数也要放在 主GPU 中

torch.nn.DataParallel(model, device_ids=[arg.local_rank], output_device=arg.local_rank)  
> 注意：s.environ[“CUDA_VISIBLE_DEVICES”] = “ ”  一定要和 device_ids 一致，且第一个为主卡，output_device 输出到主卡
> 或者上面都不写，默认使用全部GPU 


保存模型：
torch.save(net.module.state_dict(), path)

加载模型：
net=nn.DataParallel(Resnet18())
net.load_state_dict(torch.load(path))
net=net.module

优化器使用：
optimizer.step() --> optimizer.module.step()
```
