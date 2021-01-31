import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim	                  # 实现各种优化算法的包
import pdb
import os
import sys
sys.path.append("..")
from utils.utils import load_mesh,load_pretrained_model,evaluate,plot_contourf_vorticity,DatasetFromTxt,generate_data
from utils.utils import save_gradient,gradient,Compute_gradient, StatGrad,normalizeGrad,adjustGrad

global_nu=0.05

phi3=torch.sin
class FullyConnectedNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(FullyConnectedNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # input layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)          # output layer
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m,0,1)
            except:
                normal_init(block, 0, 1)

    def forward(self, x):
        x = phi3(self.hidden1(x))             # activation function for hidden layer
        x = phi3(self.hidden2(x))
        x = phi3(self.hidden3(x))
        x = phi3(self.hidden4(x))
        x = self.predict(x)                        # linear output
        return x

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def train(args, model_list, device, interior_train_loader, 
          dirichlet_bdry_training_data_loader, 
          optimizer, epoch,lamda,beta,gamma): # 还可添加loss_func等参数
    retloss=[]
    bdryiter= iter(dirichlet_bdry_training_data_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x        =Variable(data,           requires_grad=True )     
        f        =Variable(target[:, 0:2], requires_grad=False)
        divf     =Variable(target[:,   2], requires_grad=False)
        divu_RHS =Variable(target[:,   3], requires_grad=False)

        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        loss_total,bound = ResLoss_upw(x,bdry_x,f,divu_RHS,divf,bdry_velocity,beta,lamda,model_list,epoch)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        lamda_temp = lamda

        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>5d}  [{:>6d}/{} ({:3.0f}%)]  Loss of bound: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), bound.item()))
        if batch_idx==0:
            retloss=loss_total
            retbound = 0
            retres = 0
            retcoar_loss = 0

    return retloss, lamda_temp, retbound, retres, retcoar_loss
def ResLoss_upw(x,bdry_x,f,divu_RHS,divf,bdry_velocity,beta,lamda,model_list,epoch):
    
    u_predict = model_list[1](bdry_x) 
    loss_function = nn.MSELoss()
    loss4 = loss_function(u_predict, bdry_velocity[:, 0:2])
    bound = (loss4)              # 调用损失函数计算损失
    loss = bound 
    
    return loss,bound

