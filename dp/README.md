
# torch.nn.DataParallel(model)

保存模型：
torch.save(net.module.state_dict(), path)

加载模型：
net=nn.DataParallel(Resnet18())
net.load_state_dict(torch.load(path))
net=net.module

优化器使用：
optimizer.step() --> optimizer.module.step()
