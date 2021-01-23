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
import matplotlib.tri as tri
# from mayavi import mlab

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim	                  # 实现各种优化算法的包

#from gpu_memory_log import gpu_memory_log

from CustomizedDataSet import DatasetFromTxt


nu=0.1
Re=1.0/nu
lambda_const=Re/2.0-np.sqrt(Re*Re/4.0+4.0*np.pi*np.pi)

def velocity(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    cosx2=np.reshape(np.cos(2.0*np.pi*x[:, 1]), (x.shape[0], 1))
    sinx2=np.reshape(np.sin(2.0*np.pi*x[:, 1]), (x.shape[0], 1))
    yb1=1.0-expx1*cosx2
    yb2=lambda_const*expx1*sinx2/(2.0*np.pi)
    return np.concatenate((yb1, yb2), axis=1)

def pressure(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    return expx1*expx1/2.0 


# def phi3(x):
#     return (
#         1 / 2 * (F.relu(x - 0)+F.relu(x - 3))*(F.relu(x - 0)-F.relu(x - 3))
#       + 3 / 2 * (F.relu(x - 2)-F.relu(x - 1))*(F.relu(x - 2)+F.relu(x - 1))
#     )

phi3=torch.sin

class FullyConnectedNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(FullyConnectedNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # input layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)          # output layer

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
        self.scalenet1 = FullyConnectedNet(n_feature, n_hidden, 1)  # input layer
        self.scalenet2 = FullyConnectedNet(n_feature, n_hidden, 1)
        self.scalenet3 = FullyConnectedNet(n_feature, n_hidden, 1)   # hidden layer
        self.scalenet4 = FullyConnectedNet(n_feature, n_hidden, 1)
        self.scalenet5 = FullyConnectedNet(n_feature, n_hidden, 1)
        self.scalenet6 = FullyConnectedNet(n_feature, n_hidden, 1)         
        self.predict = torch.nn.Linear(6, n_output)  # output layer

    def forward(self, x):
        # activation function for hidden layer
        y1 = self.scalenet1(x)
        y2 = self.scalenet2(2*x)
        y3 = self.scalenet3(4*x)
        y4 = self.scalenet4(8*x)
        y5 = self.scalenet5(16*x)
        y6 = self.scalenet6(32*x)
        
        x=torch.cat((y1, y2, y3, y4, y5, y6), 1)
        
        x = self.predict(x)                        # linear output
        
        return x
      
def plot_velocity_error(model_u, epoch):
    X = np.arange(   0,   2, 0.01).astype(np.float32)
    Y = np.arange(-0.5, 1.5, 0.01).astype(np.float32)
    nx=len(X)
    ny=len(Y)
    X, Y = np.meshgrid(X, Y)
    X1=np.reshape(X, (nx*ny, 1))
    Y1=np.reshape(Y, (nx*ny, 1))
    
    points=np.concatenate((X1, Y1), axis=1)
    p=0
    jump=1000
    while p+jump<=len(x):
        XY=Variable(torch.tensor(points[p:p+jump, :]),requires_grad=False).to(device)
        velocity_on_gpu=model_u(XY)
        v1[p:p+jump] = velocity_on_gpu[:, 0].cpu().data.numpy()
        v2[p:p+jump] = velocity_on_gpu[:, 1].cpu().data.numpy()   
        p=p+jump
    XY=Variable(torch.tensor(points[p:, :]),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1[p:] = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2[p:] = velocity_on_gpu[:, 1].cpu().data.numpy()   
    
    v1 = np.reshape(v1, (nx, ny)) 
    v2 = np.reshape(v2, (nx, ny)) 
    
    v_exact=velocity(points)
    
    v_exact_1=np.reshape(v_exact[:, 0], (nx, ny))
    v_exact_2=np.reshape(v_exact[:, 1], (nx, ny))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    if (nx*ny>10000):
        ax.plot_surface(X, Y, v1-v_exact_1, rstride=1, cstride=1, cmap='rainbow', rasterized=True)
    else:
        ax.plot_surface(X, Y, v1-v_exact_1, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x',fontsize=14,alpha=1.0)
    ax.set_ylabel('y',fontsize=14,alpha=1.0)
    ax.set_zlabel('error',fontsize=14,alpha=1.0)
    ax.elev = 30
    ax.azim = -37.5
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_x.pdf')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    if (nx*ny>10000):
        ax.plot_surface(X, Y, v2-v_exact_2, rstride=1, cstride=1, cmap='rainbow', rasterized=True)
    else:
        ax.plot_surface(X, Y, v2-v_exact_2, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x',fontsize=14,alpha=1.0)
    ax.set_ylabel('y',fontsize=14,alpha=1.0)
    ax.set_zlabel('error',fontsize=14,alpha=1.0)
    ax.elev = 30
    ax.azim = -37.5
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_y.pdf')
    plt.close('all')
    
def plot_surf_velocity(model_u, epoch):
    X = np.arange(   0,   2, 0.01).astype(np.float32)
    Y = np.arange(-0.5, 1.5, 0.01).astype(np.float32)
    nx=len(X)
    ny=len(Y)
    X, Y = np.meshgrid(X, Y)
    X1=np.reshape(X, (nx*ny, 1))
    Y1=np.reshape(Y, (nx*ny, 1))
    
    points=np.concatenate((X1, Y1), axis=1)
    p=0
    jump=1000
    while p+jump<=len(x):
        XY=Variable(torch.tensor(points[p:p+jump, :]),requires_grad=False).to(device)
        velocity_on_gpu=model_u(XY)
        v1[p:p+jump] = velocity_on_gpu[:, 0].cpu().data.numpy()
        v2[p:p+jump] = velocity_on_gpu[:, 1].cpu().data.numpy()   
        p=p+jump
    XY=Variable(torch.tensor(points[p:, :]),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1[p:] = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2[p:] = velocity_on_gpu[:, 1].cpu().data.numpy()   
    
    v1 = np.reshape(v1, (nx, ny)) 
    v2 = np.reshape(v2, (nx, ny)) 
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    if (nx*ny>10000):
        ax.plot_surface(X, Y, v1, rstride=1, cstride=1, cmap='rainbow', rasterized=True)
    else:
        ax.plot_surface(X, Y, v1, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x',fontsize=14,alpha=1.0)
    ax.set_ylabel('y',fontsize=14,alpha=1.0)
    ax.set_zlabel('error',fontsize=14,alpha=1.0)
    ax.elev = 30
    ax.azim = -37.5
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_x.pdf')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    if (nx*ny>10000):
        ax.plot_surface(X, Y, v2, rstride=1, cstride=1, cmap='rainbow', rasterized=True)
    else:
        ax.plot_surface(X, Y, v2, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x',fontsize=14,alpha=1.0)
    ax.set_ylabel('y',fontsize=14,alpha=1.0)
    ax.set_zlabel('error',fontsize=14,alpha=1.0)
    ax.elev = 30
    ax.azim = -37.5
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_y.pdf')
    plt.close('all')
    

    
def plot_contourf_velocity(model_u, epoch):
    X = np.arange(   0,   2, 0.01).astype(np.float32)
    Y = np.arange(-0.5, 1.5, 0.01).astype(np.float32)
    nx=len(X)
    ny=len(Y)
    X, Y = np.meshgrid(X, Y)
    X1=np.reshape(X, (nx*ny, 1))
    Y1=np.reshape(Y, (nx*ny, 1))
    
    points=np.concatenate((X1, Y1), axis=1)
    p=0
    jump=1000
    while p+jump<=len(x):
        XY=Variable(torch.tensor(points[p:p+jump, :]),requires_grad=False).to(device)
        velocity_on_gpu=model_u(XY)
        v1[p:p+jump] = velocity_on_gpu[:, 0].cpu().data.numpy()
        v2[p:p+jump] = velocity_on_gpu[:, 1].cpu().data.numpy()   
        p=p+jump
    XY=Variable(torch.tensor(points[p:, :]),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1[p:] = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2[p:] = velocity_on_gpu[:, 1].cpu().data.numpy()   
    
    v1 = np.reshape(v1, (nx, ny)) 
    v2 = np.reshape(v2, (nx, ny))   
    
    plt.figure()
    plt.contourf(X, Y, v1, 10, cmap=plt.cm.Spectral)
    plt.axis('equal')
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('y',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_x_contour.pdf')
    
    plt.figure()
    plt.contourf(X, Y, v2, 10, cmap=plt.cm.Spectral)
    plt.axis('equal')
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('y',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_y_contour.pdf')
    plt.close('all')
    
def plot_velocity_along_line(model_u, epoch):
    X = np.arange(   0,   2, 0.001).astype(np.float32)
    Y = np.array([0.7]*len(X)).astype(np.float32)
    
    XY=Variable(torch.tensor(np.array([X, Y]).T),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1 = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2 = velocity_on_gpu[:, 1].cpu().data.numpy()
   
    
    v_exact=velocity(np.array([X, Y]).T)
    
    plt.figure()
    plt.plot(X[0::10], v1[0::10], 'b*')
    plt.plot(X, v_exact[:,0],  lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_x$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_x_along_line.pdf')
    
    plt.figure()
    plt.plot(X[0::10], v2[0::10], 'b*')
    plt.plot(X, v_exact[:,1], lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_y$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_y_along_line.pdf')
    plt.close('all')
    
def test(args, model_u, model_p, device, test_loader):
    model_u.eval()          # 必备，将模型设置为评估模式
    model_p.eval()          # 必备，将模型设置为评估模式
    u_test_err = 0
    p_test_err = 0
    with torch.no_grad(): # 禁用梯度计算
        for data, target in test_loader: # 从数据加载器迭代一个batch的数据
            data  =Variable(  data, requires_grad=True ).to(device)
            target=Variable(target, requires_grad=False).to(device)
            u_output = model_u(data)
            p_output = model_p(data)
            p_c=p_output[0,0]-target[0, 2]
            p_output = p_output-p_c
            loss_function = nn.MSELoss(reduction = 'sum')
            u_test_err += loss_function(u_output, target[:, 0:2])    # sum up batch loss
            p_test_err += loss_function(p_output, target[:, 2:3])    # sum up batch loss
            
    u_test_err = np.sqrt(u_test_err.cpu()/ len(test_loader.dataset))
    p_test_err = np.sqrt(p_test_err.cpu()/ len(test_loader.dataset))
    print('\nTest set: l^2 error of velocity on test points: {:.8f}\n'.format(u_test_err))
    print('\nTest set: l^2 error of pressure on test points: {:.8f}\n'.format(p_test_err))
    return u_test_err, p_test_err


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    
    nNeuron=50
    model_u = MultiScaleNet(2, nNeuron, 2).to(device)	# 实例化自定义网络模型
    model_p = MultiScaleNet(2, nNeuron, 1).to(device)
    test_dataset=DatasetFromTxt('test_data.dat')
    
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, **kwargs)
    l2err_of_u=[]
    l2err_of_p=[]
    for epoch in range(50, 350, 50):
        model_u.load_state_dict(torch.load('netsave/u_net_params_at_epochs'+str(epoch)+'.pkl'))
        model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(epoch)+'.pkl'))
        u_err, p_err=test(args, model_u, model_p, device, test_data_loader)
        l2err_of_u.append(u_err.item())
        l2err_of_p.append(p_err.item())
        
    np.savetxt('result_plots/l2error_of_velocity'+str(0)+'to'+str(2000)+'.txt', l2err_of_u, fmt="%f", delimiter=",")
    np.savetxt('result_plots/l2error_of_pressure'+str(0)+'to'+str(2000)+'.txt', l2err_of_p, fmt="%f", delimiter=",")
    
    
    
# if __name__ == '__main__':
    
#     device = torch.device("cuda") # 设置使用CPU or GPU
#     nNeuron=50
#     model_u = MultiScaleNet(2, nNeuron, 2).to(device)	# 实例化自定义网络模型
#     model_p = FullyConnectedNet(2, nNeuron, 1).to(device)	# 实例化自定义网络模型
    
#     epoch=2000
    
#     model_u.load_state_dict(torch.load('netsave/u_net_params_at_epochs'+str(epoch)+'.pkl'))
#     model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(epoch)+'.pkl'))
# #     plot_velocity_error(model_u,  epoch) 
    
# #     plot_velocity_along_line(model_u, epoch)
#     plot_contourf_velocity(model_u, epoch)
     