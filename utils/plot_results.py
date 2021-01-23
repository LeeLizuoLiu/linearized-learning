import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import matplotlib
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim	                  # 实现各种优化算法的包
import pdb
import os
#from gpu_memory_log import gpu_memory_log

from CustomizedDataSet import DatasetFromTxt
from generate_data_in_2D_domain import *

global_nu=0.01

# def phi3(x):
#     return (
#         1 / 2 * (F.relu(x - 0)+F.relu(x - 3))*(F.relu(x - 0)-F.relu(x - 3))
#       + 3 / 2 * (F.relu(x - 2)-F.relu(x - 1))*(F.relu(x - 2)+F.relu(x - 1))
#     )



phi3=torch.sin


def load_mesh(datafile):
    with open(datafile) as f:
        lines = f.readlines()
        Npoint, Nelement, poly = np.loadtxt(lines[0:1], int)
        points = np.loadtxt(lines[1: 1+Npoint], np.float32)
        elements = np.loadtxt(lines[1+Npoint: (1+Npoint+Nelement)], int)-1
    return points, elements

def plot_contourf_vorticity(model_w,model_u, epoch, points,elements,XY):
    x, y = points.T
    triangulation = tri.Triangulation(x, y, elements)
    
    velocity = model_u(XY)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    tcf = ax.tricontourf(triangulation, velocity[:,0].cpu().data.numpy(), cmap='rainbow',levels=100)
    fig.colorbar(tcf)
    ax.axis('equal')
    fig.savefig('result_plots/Epoch_'+str(epoch)+'velocity_contour.pdf')
    plt.close()

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

class phi(nn.Module):
    def __init__(self):
        super(phi,self).__init__()
    def forward(self,x):
        return torch.sin(x)

class MultiScaleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size = 32, nb_heads=1):
        super(MultiScaleNet, self).__init__()
#        self.network = nn.Sequential(
#            nn.Conv1d(in_channels=input_dim * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
#            phi(),
#            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
#            phi(),
#            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
#            phi(),
#            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
#            phi(),
#            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=1 * nb_heads, kernel_size=1, groups=nb_heads),
#            nn.Conv1d(in_channels=1 * nb_heads, out_channels=output_dim, kernel_size=1)
#        )
#        self.nb_heads = nb_heads
#        self.register_buffer('scale',torch.Tensor([[1.5**(j) for i in range(input_dim)] for j in range(nb_heads)]).view(input_dim*nb_heads,1))
#      
#    def forward(self, x):
#      x = x.repeat(1, self.nb_heads).unsqueeze(-1)
#      x = x*self.scale
#      flat = self.network(x)
#      return torch.squeeze(flat)


        self.scalenet01 = FullyConnectedNet(input_dim, hidden_size, 1)  # input layer
        self.scalenet02 = FullyConnectedNet(input_dim, hidden_size, 1)
        self.scalenet03 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet04 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet05 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet06 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet07 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet08 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet09 = FullyConnectedNet(input_dim, hidden_size, 1)  # input layer
        self.scalenet10 = FullyConnectedNet(input_dim, hidden_size, 1)  # input layer
        self.scalenet11 = FullyConnectedNet(input_dim, hidden_size, 1)  # input layer
        self.scalenet12 = FullyConnectedNet(input_dim, hidden_size, 1)
        self.scalenet13 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet14 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet15 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet16 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet17 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet18 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.predict = torch.nn.Linear(8, output_dim)  # output layer
    def forward(self, x):
       # activation function for hidden layer
        alpha = 2 
        y01 = self.scalenet01(x)
        y03 = self.scalenet03(alpha**1.0*x)
        y05 = self.scalenet05(alpha**2.0*x)
        y07 = self.scalenet07(alpha**3.0*x)
        y09 = self.scalenet09(alpha**4.0*x)
        y11 = self.scalenet11(alpha**5.0*x)
        y13 = self.scalenet13(alpha**6.0*x)
        y15 = self.scalenet15(alpha**7.0*x)
        x=torch.cat((y01,y03,y05,y07,y09,y11,y13,y15), 1)
        x = self.predict(x)                        # linear output
        return x


def evaluate(model_u,device,epoch):
    if os.path.isfile('Data/velocity_x@__57.txt'):
       yu = np.loadtxt('Data/velocity_x@__57.txt').astype(np.float32)
       u = yu[:,1]
       x = np.ones_like(yu)*-0.57
       x[:,1] = yu[:,0]
       predict = model_u(torch.Tensor(x).to(device)).cpu().data.numpy()
       err = abs(predict[:,0]-u)/np.max(u)
       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       tcf = ax.plot(yu[:,0],err)
       fig.savefig('result_plots/err057_u'+str(epoch)+'.pdf')
       plt.close()

       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       ax.plot(yu[:,0],u)
       ax.plot(yu[:,0],predict[:,0])
       fig.savefig('result_plots/u057'+str(epoch)+'.pdf')
       plt.close()




if __name__ == '__main__':
    # Training settings
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--nbatch', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5456, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU
    device = torch.device("cuda:0" if use_cuda else "cpu") # 设置使用CPU or GPU
    
    nNeuron=100
    nb_head = 18

    model_u = MultiScaleNet(2, 2, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_w = MultiScaleNet(2, 4, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_p = MultiScaleNet(2, 1, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型

    matplotlib.use("Agg")
    epoch=50000

    model_u.load_state_dict(torch.load('netsave/u_net_params_at_epochs'+str(epoch)+'.pkl'))
    model_w.load_state_dict(torch.load('netsave/w_net_params_at_epochs'+str(epoch)+'.pkl'))
    model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(epoch)+'.pkl'))
    
    points, elements = load_mesh("mesh_file.dat")
    XY=Variable(torch.tensor(points),requires_grad=True).to(device)
    
    evaluate(model_u,device,epoch)