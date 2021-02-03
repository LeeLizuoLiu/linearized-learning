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
sys.path.append("..")
from utils.utils import FullyConnectedNet,load_mesh,load_pretrained_model,evaluate,plot_contourf_vorticity,DatasetFromTxt,generate_data_dirichlet,StatGrad,data_generator
from NS_msnn import MultiScaleNet,train,global_nu

nu=0.05
Re=1.0/nu
lambda_const=Re/2.0-np.sqrt(Re*Re/4.0+4.0*np.pi*np.pi)
n_freq=40
m_freq=45

def velocity(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    cosx1x2=np.reshape(np.cos(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    sinx1x2=np.reshape(np.sin(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    yb1=1.0-expx1*cosx1x2
    yb2=lambda_const*expx1*sinx1x2/(2.0*m_freq*np.pi)+n_freq/m_freq*expx1*cosx1x2
    return np.concatenate((yb1, yb2), axis=1)

def pressure(x):
    expx1=1/2*(1-np.exp(2*lambda_const*x[:,0:1]))
    return expx1 


# def phi3(x):
#     return (
#         1 / 2 * (F.relu(x - 0)+F.relu(x - 3))*(F.relu(x - 0)-F.relu(x - 3))
#       + 3 / 2 * (F.relu(x - 2)-F.relu(x - 1))*(F.relu(x - 2)+F.relu(x - 1))
#     )

phi3=torch.sin

from NS_msnn import MultiScaleNet

def load_mesh(datafile):
    with open(datafile) as f:
        lines = f.readlines()
        Npoint, Nelement, poly = np.loadtxt(lines[0:1], int)
        points = np.loadtxt(lines[1: 1+Npoint], np.float32)
        elements = np.loadtxt(lines[1+Npoint: (1+Npoint+Nelement)], int)-1
    return points, elements


# mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(800, 800))
# points, elements = load_mesh("mesh_file.dat")
# x, y = points.T
# z = np.zeros_like(x)
# mlab.triangular_mesh(x, y, z, elements, representation="mesh", tube_radius=0.001)
# mlab.triangular_mesh(x, y, f([x,y]), elements)
# mlab.savefig("mesh_file2.pdf")

        
def plot_velocity_error(model_u, epoch):
    points, elements = load_mesh("mesh_file_1.dat")
    x, y = points.T
    triangulation = tri.Triangulation(x, y, elements)
    v1=np.zeros(len(x))
    v2=np.zeros(len(x))
    
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
    
    v_exact=velocity(points)
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    ax.plot_trisurf(triangulation, v1-v_exact[:,0], cmap='rainbow')
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_x.pdf')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.patch.set_alpha(1.0)
    ax.plot_trisurf(triangulation, v2-v_exact[:,1], cmap='rainbow')
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_y.pdf')
    
def plot_surf_velocity(model_u, epoch):
    points, elements = load_mesh("mesh_file_1.dat")
    x, y = points.T
    triangulation = tri.Triangulation(x, y, elements)
    v1=np.zeros(len(x))
    v2=np.zeros(len(x))
    
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
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(triangulation, v1, cmap='rainbow')
    ax.patch.set_alpha(1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_x.pdf')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(triangulation, v2, cmap='rainbow')
    ax.patch.set_alpha(1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_y.pdf')
    
def plot_contourf_velocity(model_u, epoch):
    points, elements = load_mesh("mesh_file_1.dat")
    x, y = points.T
    triangulation = tri.Triangulation(x, y, elements)
    v1=np.zeros(len(x))
    v2=np.zeros(len(x))
    
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
    
    plt.figure()
    plt.tricontourf(triangulation, v1, cmap='rainbow', rasterized=True, bbox_inches='tight')
    plt.axis('equal')
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('y',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_x_contour.pdf')
    
    plt.figure()
    plt.tricontourf(triangulation, v2, cmap='rainbow', rasterized=True, bbox_inches='tight')
    plt.axis('equal')
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('y',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'velocity_y_contour.pdf')
    
def plot_velocity_along_line(model_u, epoch):
    X = np.arange(   0.0,   2.0, 0.0002).astype(np.float32)
    Y = np.array([0.7]*len(X)).astype(np.float32)
    xy=np.array([X, Y]).T
    XY=Variable(torch.tensor(xy),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1 = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2 = velocity_on_gpu[:, 1].cpu().data.numpy()
   
    
    v_exact=velocity(xy)
    
    plt.figure()
    plt.plot(X, v_exact[:,0],  lw=1)
    plt.plot(X, v1,   lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_x$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'_velocity_x_along_line.pdf')
    
    plt.figure()
    plt.plot(X, v_exact[:,1], lw=1)
    plt.plot(X, v2,   lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_y$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'_velocity_y_along_line.pdf')
    
    print(v1)
    v=np.vstack((v1, v2)).T
    np.savetxt('result_plots/exact_velocity'+'.txt', v_exact, fmt="%f", delimiter=",")
    np.savetxt('result_plots/MSDNN_VGVP_velocity'+str(epoch)+'.txt', v, fmt="%f", delimiter=",")
    
def plot_pressure_along_line(model_p, epoch):
    model_p.eval() 
    X = np.arange(   0.0,   2.0, 0.0002).astype(np.float32)
    Y = np.array([0.8]*len(X)).astype(np.float32)
    
    XY=Variable(torch.tensor(np.array([X, Y]).T),requires_grad=False).to(device)
    pressure_on_gpu=model_p(XY)
    p = pressure_on_gpu.cpu().data.numpy()
    p_exact=pressure(np.array([X, Y]).T)
    p_c = p[0]-p_exact[0]
    p = p - p_c
    ftsize=14
    print(np.mean(p_exact-p))
    plt.figure()
    plt.plot(X, p_exact, label='exact', lw=1)
    plt.plot(X[0::150], p[0::150], 'r*', label='MSDNN')
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('p',fontsize=14,alpha=1.0)
    plt.tick_params(labelsize=ftsize)
    plt.legend(fontsize=ftsize,loc="upper right")
    plt.savefig('result_plots/Epoch'+str(epoch)+'_pressure_x_along_line.pdf')
    
    np.savetxt('result_plots/exact_pressure'+'.txt', p_exact, fmt="%f", delimiter=",")
    np.savetxt('result_plots/MSDNN_VGVP_pressure'+str(epoch)+'.txt', p, fmt="%f", delimiter=",")
    
def plot_velocity_along_line_error(model_u, epoch):
    X = np.arange(   0.0,   2.0, 0.0002).astype(np.float32)
    Y = np.array([0.7]*len(X)).astype(np.float32)
    xy=np.array([X, Y]).T
    XY=Variable(torch.tensor(xy),requires_grad=False).to(device)
    velocity_on_gpu=model_u(XY)
    v1 = velocity_on_gpu[:, 0].cpu().data.numpy()
    v2 = velocity_on_gpu[:, 1].cpu().data.numpy()
   
    
    v_exact=velocity(xy)
    
    plt.figure()
    plt.plot(X, np.abs(v_exact[:,0]-v1)/np.max(v_exact[:,0]),  lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_x$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_x_along_line.pdf')
    
    plt.figure()
    plt.plot(X, np.abs(v_exact[:,1]-v2)/np.max(v_exact[:,1]), lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('$v_y$',fontsize=14,alpha=1.0)
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_velocity_y_along_line.pdf')
    
    
def plot_pressure_along_line_error(model_p, epoch):
    model_p.eval() 
    X = np.arange(   0.0,   2.0, 0.0002).astype(np.float32)
    Y = np.array([0.8]*len(X)).astype(np.float32)
    
    XY=Variable(torch.tensor(np.array([X, Y]).T),requires_grad=False).to(device)
    pressure_on_gpu=model_p(XY)
    p = pressure_on_gpu.cpu().data.numpy()
    p_exact=pressure(np.array([X, Y]).T)
    p_c = p[0]-p_exact[0]
    p = p - p_c
    ftsize=14
    print(np.mean(np.abs(p_exact-p))/np.max(p_exact))
    plt.figure()
    plt.plot(X, np.abs(p-p_exact)/np.max(p_exact), label='exact', lw=1)
    plt.xlabel('x',fontsize=14,alpha=1.0)
    plt.ylabel('p',fontsize=14,alpha=1.0)
    plt.tick_params(labelsize=ftsize)
    plt.legend(fontsize=ftsize,loc="upper right")
    plt.savefig('result_plots/Epoch'+str(epoch)+'error_of_pressure_x_along_line.pdf')
 
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


# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for training (default: 500)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
    
#     args = parser.parse_args()
    
#     use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU

#     torch.manual_seed(args.seed) # 设置随机种子，保证可重复性

#     device = torch.device("cuda" if use_cuda else "cpu") # 设置使用CPU or GPU

#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    
#     nNeuron=50
#     model_u = MultiScaleNet(9, 2, nNeuron, 2).to(device)	# 实例化自定义网络模型
#     model_p = FullyConnectedNet(phi3, 2, nNeuron, 1).to(device)
#     test_dataset=DatasetFromTxt('test_data.dat')
    
#     test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, **kwargs)
#     l2err_of_u=[]
#     l2err_of_p=[]
#     for epoch in range(50, 1501, 50):
#         model_u.load_state_dict(torch.load('netsave/u_net_params_at_epochs'+str(epoch)+'.pkl'))
#         model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(epoch)+'.pkl'))
#         u_err, p_err=test(args, model_u, model_p, device, test_data_loader)
#         l2err_of_u.append(u_err.item())
#         l2err_of_p.append(p_err.item())
        
#     np.savetxt('result_plots/l2error_of_velocity'+str(50)+'to'+str(1500)+'.txt', l2err_of_u, fmt="%f", delimiter=",")
#     np.savetxt('result_plots/l2error_of_pressure'+str(50)+'to'+str(1500)+'.txt', l2err_of_p, fmt="%f", delimiter=",")
    
    
    
if __name__ == '__main__':
    
    device = torch.device("cuda") # 设置使用CPU or GPU
    nNeuron=100
    nb_head = 1
    model_p =FullyConnectedNet(2,nNeuron,1).to(device)	 #FullyConnectedNet(2,nNeuron, 1).to(device)	# 实例化自定义网络模型
    epoch=904
    model_p.load_state_dict(torch.load('netsave/p_net_params_at_epochs'+str(epoch)+'.pkl'))
    model_u = MultiScaleNet(2, 2, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_u.load_state_dict(torch.load('netsave/old_u_net_params_at_epochs'+str(epoch)+'.pkl'))
 
#     plot_velocity_error(model_u,  epoch) 
    
    plot_pressure_along_line_error(model_p, epoch)
    plot_velocity_along_line_error(model_u, epoch)
    
#     plot_surf_velocity(model_u, epoch)
#     startepochs=0
#     endepochs=1500
#     l2err_u=np.loadtxt('result_plots/l2error_of_velocity'+str(startepochs)+'to'+str(endepochs)+'.txt', np.float32)
#     l2err_p=np.loadtxt('result_plots/l2error_of_pressure'+str(startepochs)+'to'+str(endepochs)+'.txt', np.float32)
#     print(l2err_u)
#     print(l2err_p)
    
     
