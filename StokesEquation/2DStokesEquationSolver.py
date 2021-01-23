# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:40:35 2020

@author: 47584359
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:33:57 2020

@author: 47584359
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim	                  # 实现各种优化算法的包

from CustomizedDataSet import DatasetFromTxt
from generate_data_in_2D_domain import generate_data
global_nu=0.1

#def phi3(x):
#    return (
#        1 / 2 * F.relu(x - 0) ** 2
#        - 3 / 2 * F.relu(x - 1) ** 2
#        + 3 / 2 * F.relu(x - 2) ** 2
#        - 1 / 2 * F.relu(x - 3) ** 2
#    )
    
# def phi3(x):
#     return (
#         1 / 2 * (F.relu(x - 0)+F.relu(x - 3))*(F.relu(x - 0)-F.relu(x - 3))
#       + 3 / 2 * (F.relu(x - 2)-F.relu(x - 1))*(F.relu(x - 2)+F.relu(x - 1))
#     )

phi3=torch.sin

class FullyConnectedNet(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(FullyConnectedNet, self).__init__()
            
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # input layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, 1)          # output layer

    def forward(self, x):
        x = phi3(self.hidden1(x))             # activation function for hidden layer
        x = phi3(self.hidden2(x))
        x = phi3(self.hidden3(x))
        x = phi3(self.hidden4(x))
        x = self.predict(x)                        # linear output
        return x

class MultiScaleNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MultiScaleNet, self).__init__()
            
        self.scalenet1 = FullyConnectedNet(n_feature, n_hidden)  # input layer
        self.scalenet2 = FullyConnectedNet(n_feature, n_hidden)
        self.scalenet3 = FullyConnectedNet(n_feature, n_hidden)   # hidden layer
        self.scalenet4 = FullyConnectedNet(n_feature, n_hidden)
        self.scalenet5 = FullyConnectedNet(n_feature, n_hidden)
        self.scalenet6 = FullyConnectedNet(n_feature, n_hidden)          
        self.scalenet7 = FullyConnectedNet(n_feature, n_hidden)
        self.scalenet8 = FullyConnectedNet(n_feature, n_hidden)
        self.scalenet9 = FullyConnectedNet(n_feature, n_hidden)          
        self.predict = torch.nn.Linear(9, n_output)  # output layer

    def forward(self, x):
        y1 = self.scalenet1(  x)             # activation function for hidden layer
        y2 = self.scalenet2(2*x)
        y3 = self.scalenet3(4*x)
        y4 = self.scalenet4(8*x)
        y5 = self.scalenet5(16*x)
        y6 = self.scalenet6(32*x)
        y7 = self.scalenet7(2**6*x)
        y8 = self.scalenet8(2**7*x)
        y9 = self.scalenet9(2**8*x)
        x=torch.cat((y1, y2, y3, y4, y5, y6,y7,y8,y9), 1)
        
        x = self.predict(x)                        # linear output
        
        return x
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(args, model_list, device, beta, interior_train_loader, bdry_train_loader, optimizer, epoch): # 还可添加loss_func等参数
    for i in range(4):
        model_list[i].train()  
    retloss=[]                                                       

    bdryiter=iter(bdry_train_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x    =Variable(data,           requires_grad=True )     
        f    =Variable(target[:, 0:2], requires_grad=False)
        divf =Variable(target[:,   2], requires_grad=False)
        
        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        
        loss_u = loss_of_u(x,bdry_x,f,divf,bdry_velocity,data)
        optimizer[0].zero_grad()                             # 清除所有优化的梯度
        loss_u.backward() # 反向传播
        optimizer[0].step() # 更新参数
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>10d}\t\t[{:>10d}/{} ({:3.0f}%)]\tLoss of u: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_u.item()))
 
    bdryiter=iter(bdry_train_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x    =Variable(data,           requires_grad=True )     
        f    =Variable(target[:, 0:2], requires_grad=False)
        divf =Variable(target[:,   2], requires_grad=False)
        
        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        
        loss_p = loss_of_p(x,bdry_x,f,divf,bdry_velocity,data)
        optimizer[1].zero_grad()                             # 清除所有优化的梯度
        loss_p.backward() # 反向传播
        optimizer[1].step() # 更新参数
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>10d}\t\t[{:>10d}/{} ({:3.0f}%)]\tLoss of p: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_p.item()))
 
    bdryiter=iter(bdry_train_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x    =Variable(data,           requires_grad=True )     
        f    =Variable(target[:, 0:2], requires_grad=False)
        divf =Variable(target[:,   2], requires_grad=False)
        
        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        
        loss_w = loss_of_w(x,bdry_x,f,divf,bdry_velocity,data)
        optimizer[2].zero_grad()                             # 清除所有优化的梯度
        loss_w.backward() # 反向传播
        optimizer[2].step() # 更新参数
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>10d}\t\t[{:>10d}/{} ({:3.0f}%)]\tLoss of w: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_w.item()))
 
    bdryiter=iter(bdry_train_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x    =Variable(data,           requires_grad=True )     
        f    =Variable(target[:, 0:2], requires_grad=False)
        divf =Variable(target[:,   2], requires_grad=False)
        
        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        
        loss_q = loss_of_q(x,bdry_x,f,divf,bdry_velocity,data)
        optimizer[3].zero_grad()                             # 清除所有优化的梯度
        loss_q.backward() # 反向传播
        optimizer[3].step() # 更新参数

        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>10d}\t\t[{:>10d}/{} ({:3.0f}%)]\tLoss of q: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_q.item()))
        if batch_idx==0:
            retloss=0
    return retloss

def loss_of_u(x,bdry_x,f,divf,bdry_velocity,data):
    interior_w_predict = model_list[2](x) 
    interior_u_predict = model_list[0](x) 
    
    interior_q_predict = model_list[3](x)        
    bdry_u_predict     = model_list[0](bdry_x)
 
    grad_u1 = torch.autograd.grad(interior_u_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 0])])
    grad_u2 = torch.autograd.grad(interior_u_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 1])])
    
    grad_w11= torch.autograd.grad(interior_w_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 0])])
    grad_w12= torch.autograd.grad(interior_w_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 1])])
    grad_w21= torch.autograd.grad(interior_w_predict[:, 2], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 2])])
    grad_w22= torch.autograd.grad(interior_w_predict[:, 3], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 3])])
    

    divu =grad_u1[0][:,0]+grad_u2[0][:,1]
    divw1=grad_w11[0][:, 0]+grad_w12[0][:, 1]
    divw2=grad_w21[0][:, 0]+grad_w22[0][:, 1]
    
    loss_function = nn.MSELoss()
    
    loss1 = loss_function(-global_nu*divw1+interior_q_predict[:,0], f[:,0])
    loss2 = loss_function(-global_nu*divw2+interior_q_predict[:,1], f[:,1])
    
    loss3 = loss_function(grad_u1[0], interior_w_predict[:,0:2])+loss_function(grad_u2[0], interior_w_predict[:, 2:4])
    loss6 = loss_function(bdry_u_predict, bdry_velocity[:, 0:2])
    loss7 = loss_function(divu, torch.tensor([0.0]*len(data)).to(device))
    loss  = loss1+loss2+loss3+100.0*loss6+loss7              # 调用损失函数计算损失
    return loss
 
def loss_of_p(x,bdry_x,f,divf,bdry_velocity,data):
    interior_p_predict = model_list[1](x) 
    interior_q_predict = model_list[3](x)        
    
    grad_p = torch.autograd.grad(interior_p_predict, x,  create_graph=True, grad_outputs=[torch.ones_like(interior_p_predict)])
    
    loss_function = nn.MSELoss()
    
    loss5 = loss_function(grad_p[0], interior_q_predict)
    loss  = loss5              # 调用损失函数计算损失
    return loss
 
def loss_of_w(x,bdry_x,f,divf,bdry_velocity,data):
    interior_w_predict = model_list[2](x) 
    interior_u_predict = model_list[0](x) 
    
    interior_q_predict = model_list[3](x)        
    bdry_u_predict     = model_list[0](bdry_x)
 
    grad_u1 = torch.autograd.grad(interior_u_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 0])])
    grad_u2 = torch.autograd.grad(interior_u_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 1])])
    
    grad_w11= torch.autograd.grad(interior_w_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 0])])
    grad_w12= torch.autograd.grad(interior_w_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 1])])
    grad_w21= torch.autograd.grad(interior_w_predict[:, 2], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 2])])
    grad_w22= torch.autograd.grad(interior_w_predict[:, 3], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 3])])
    
    divw1=grad_w11[0][:, 0]+grad_w12[0][:, 1]
    divw2=grad_w21[0][:, 0]+grad_w22[0][:, 1]
    
    loss_function = nn.MSELoss()
    
    loss1 = loss_function(-global_nu*divw1+interior_q_predict[:,0], f[:,0])
    loss2 = loss_function(-global_nu*divw2+interior_q_predict[:,1], f[:,1])
    loss3 = loss_function(grad_u1[0], interior_w_predict[:,0:2])+loss_function(grad_u2[0], interior_w_predict[:, 2:4])
    loss6 = loss_function(bdry_u_predict, bdry_velocity[:, 0:2])
    loss  = loss1+loss2+loss3+100.0*loss6              # 调用损失函数计算损失
    return loss
 
def loss_of_q(x,bdry_x,f,divf,bdry_velocity,data):
    interior_w_predict = model_list[2](x) 
    interior_p_predict = model_list[1](x) 
    interior_q_predict = model_list[3](x)        
    
    grad_p = torch.autograd.grad(interior_p_predict, x,  create_graph=True, grad_outputs=[torch.ones_like(interior_p_predict)])
    
    grad_w11= torch.autograd.grad(interior_w_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 0])])
    grad_w12= torch.autograd.grad(interior_w_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 1])])
    grad_w21= torch.autograd.grad(interior_w_predict[:, 2], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 2])])
    grad_w22= torch.autograd.grad(interior_w_predict[:, 3], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_w_predict[:, 3])])
    
    grad_q1 = torch.autograd.grad(interior_q_predict[:,0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_q_predict[:,0])])
    grad_q2 = torch.autograd.grad(interior_q_predict[:,1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_q_predict[:,1])])

    divw1=grad_w11[0][:, 0]+grad_w12[0][:, 1]
    divw2=grad_w21[0][:, 0]+grad_w22[0][:, 1]
    divq =grad_q1[0][:, 0]+grad_q2[0][:, 1]
    
    loss_function = nn.MSELoss()
    
    loss1 = loss_function(-global_nu*divw1+interior_q_predict[:,0], f[:,0])
    loss2 = loss_function(-global_nu*divw2+interior_q_predict[:,1], f[:,1])
    
    loss4 = loss_function(divq, divf)
    loss5 = loss_function(grad_p[0], interior_q_predict)
    loss  = loss1+loss2+beta*loss4+loss5              # 调用损失函数计算损失
    return loss
 
   
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--nbatch', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU

    torch.manual_seed(args.seed) # 设置随机种子，保证可重复性

    device = torch.device("cuda" if use_cuda else "cpu") # 设置使用CPU or GPU

    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    
    nNeuron=100
    
    model_u = MultiScaleNet(2, nNeuron, 2).to(device)	# 实例化自定义网络模型
    model_w = MultiScaleNet(2, nNeuron, 4).to(device)	# 实例化自定义网络模型
    model_p = MultiScaleNet(2, nNeuron, 1).to(device)	# 实例化自定义网络模型
    model_q = MultiScaleNet(2, nNeuron, 2).to(device)	# 实例化自定义网络模型
    
    loadepochs=0
    
    if loadepochs!=0:
        model_u.load_state_dict(torch.load('netsave/u_net_params_at_epochs'+str(loadepochs)+'.pkl'))
        model_w.load_state_dict(torch.load('netsave/w_net_params_at_epochs'+str(loadepochs)+'.pkl'))
        model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(loadepochs)+'.pkl'))
        model_q.load_state_dict(torch.load('netsave/q_net_params_at_epochs'+str(loadepochs)+'.pkl'))
        
    params_u = model_u.parameters()
    params_p = model_p.parameters() 
    params_w = model_w.parameters()
    params_q = model_q.parameters()

    optimizer_u = optim.Adam(params_u, lr=args.lr) # 实例化求解器
    optimizer_p = optim.Adam(params_p, lr=args.lr) # 实例化求解器
    optimizer_w = optim.Adam(params_w, lr=args.lr) # 实例化求解器
    optimizer_q = optim.Adam(params_q, lr=args.lr) # 实例化求解器

    loss=np.array([0.0]*(args.epochs-loadepochs))
    model_list=[model_u,  model_p, model_w, model_q]
    optimizer_list = [optimizer_u,optimizer_p,optimizer_w,optimizer_q]
    l2err_of_u=[]
    l2err_of_p=[]  
    beta_history=[]
    beta=1
    beta_history.append(beta)
    for epoch in range(loadepochs+1, args.epochs + 1): # 循环调用train() and test()进行epoch迭代
        if  epoch%1==0 :
            x, target, xb,yb = generate_data()
            interior_training_dataset=torch.utils.data.TensorDataset(torch.Tensor(x).to(device),torch.Tensor(target).to(device))
            interior_training_data_loader = torch.utils.data.DataLoader(interior_training_dataset, 
                                                                        batch_size=int(len(interior_training_dataset)/args.nbatch),
                                                                        shuffle=True, 
                                                                        **kwargs)

            boundary_training_dataset= torch.utils.data.TensorDataset(torch.Tensor(xb).to(device),torch.Tensor(yb).to(device))
            boundary_training_data_loader =torch.utils.data.DataLoader(boundary_training_dataset,
                                                                                batch_size=int(len(boundary_training_dataset)/args.nbatch), 
                                                                                shuffle=True, 
                                                                                **kwargs)

        loss[epoch-loadepochs-1]=train(args, model_list, device, beta, interior_training_data_loader, boundary_training_data_loader, optimizer_list, epoch)
        if epoch%100==0:
            torch.save(model_u.state_dict(), 'netsave/u_net_params_at_epochs'+str(epoch)+'.pkl') 
            torch.save(model_p.state_dict(), 'netsave/p_net_params_at_epochs'+str(epoch)+'.pkl') 
            torch.save(model_w.state_dict(), 'netsave/w_net_params_at_epochs'+str(epoch)+'.pkl')
            torch.save(model_q.state_dict(), 'netsave/q_net_params_at_epochs'+str(epoch)+'.pkl') 
    
    np.savetxt('result_plots/loss'+str(loadepochs)+'to'+str(args.epochs)+'.txt', loss, fmt="%f", delimiter=",")
    
        