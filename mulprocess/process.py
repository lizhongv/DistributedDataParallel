#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Email: zl2272001@gmail.com
@Time: 2022/11/4
@target: using multiprocessing to create multi-process
"""
import os
import time
from multiprocessing import Process, Pool

"""
# 多进程：主进程（父进程）产生至少一个子进程
#1. spawn 父进程启动一个新的Python解释进程
## 子进程只会继承那些运行进程对象的run()方法所需的资源，特别是父进程中非必须的文件描述符和句柄不会被继承。
## 相对于使用fork或者forkserver，该方法启动进程相当慢
## 可在Unix和Windows上使用。Windows上的默认设置。

# 2. fork 父进程使用os.fork()来产生Python解释器分叉
## 子进程在开始时实际上与父进程相同
## 父进程的所有资源都由子进程继承
## 只存在于Unix. Unix中的默认值。

# 3. forsever 将启动服务器进程，每当需要启动一个新进程，父进程就会链接服务器并请求它分叉一个新进程
## 分叉服务器进程是单线程，使用os.fork()是安全的。
## 没有不必要的资源被继承
## 可在Unix平台上使用，支持通过Unix管道传递文件描述符

常用的方法和属性：
run():
start():
join():


"""

def func(s, num):
    # 输出传入的参数，当前子进程的ID，当前父进程的ID
    pid = os.getpid()
    ppid = os.getppid()
    print(f"{s}: {num} child process {pid}, parent process {ppid}")
    time.sleep(3)

def simple_process():
    # 打印当前进程的ID（父进程）
    print(f'main process {os.getpid()} start...')

    # 创建进程对象,使用默认类
    # func是子进程执行的函数
    p = Process(target=func, args=('Process', 0))
    # 开始运行新的进程
    print('child process start...')
    p.start()
    # 阻塞主进程，等待子进程运行完毕，再往下执行
    p.join()

    print(f'main {os.getpid()} process end!')

# 继承于Process，修改run()
class MyProcess(Process):
    def __init__(self, num):
        super(MyProcess, self).__init__()
        self.num = num

    def run(self):
        func('MyProcess', self.num)

def second_process():
    print(f'main process {os.getpid()} start...')

    p = MyProcess(0)
    print('child process start...')
    p.start()
    p.join()

    print(f'main process {os.getpid()} end!')


def pool_process():
    '''进程池，维护若干个进程
       有apply_async(异步）和apple（同步）两种方法
    '''  
    print(f'main process {os.getpid()} start...')
    start_time = time.time()  
    size = 4
    pool = Pool(size) # create 4 processions

    print('child process start...')
    for i in range(size):
        # pool.apply(func, ('apple', i))
        # apple 是阻塞的，当前子进程执行完毕后，再执行下一个子进程
        pool.apply_async(func, ('apply_async', i))
        # applY_async 是非阻塞的，主进程和子进程同时跑，谁跑的快谁先来
    pool.close()  # 关闭进程池，表示不再往进程池中添加进程，需要在join之前调用
    pool.join() # 等待进程池中的所有进程执行完毕
    print(f'main process {os.getpid()} end! total time is {time.time()-start_time}')
    
if __name__ == "__main__":
    # simple_process()
    # second_process()
    pool_process()



