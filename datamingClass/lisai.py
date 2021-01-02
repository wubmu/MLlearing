# from tensorboardX import SummaryWriter #SummaryWriter Encapsultes everything
# log_dir = "/home/user-xc/wyb/mllearing/datamingClass"
# writer = SummaryWriter(log_dir) #实例化对象时指定存放log的目录
# for i in range(10):
#     writer.add_scalar('quadratic', i**2, global_step=i)
#     writer.add_scalar('exponential', 2**i, global_step=i)
import tensorflow as tf
# print(tf.test.is_gpu_available())
# coding=gbk
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)