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
from utils.utils import FullyConnectedNet,load_mesh,load_pretrained_model,evaluate,plot_contourf_vorticity,DatasetFromTxt,generate_data
from utils.utils import save_gradient,gradient,Compute_gradient, StatGrad,normalizeGrad,adjustGrad

global_nu=0.1

class MultiScaleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size = 32, nb_heads=1):
        super(MultiScaleNet, self).__init__()
        self.scalenet01 = FullyConnectedNet(input_dim, hidden_size, 1)  # input layer
        self.scalenet02 = FullyConnectedNet(input_dim, hidden_size, 1)
        self.scalenet03 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet04 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet05 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet06 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet07 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.scalenet08 = FullyConnectedNet(input_dim, hidden_size, 1)   # hidden layer
        self.predict = torch.nn.Linear(8, output_dim)          # output layer

    def forward(self, x):
       # activation function for hidden layer
        alpha = 1 
        y01 = self.scalenet01(x)
        y02 = self.scalenet02(alpha**1.0*x)
        y03 = self.scalenet03(alpha**2.0*x)
        y04 = self.scalenet04(alpha**3.0*x)
        y05 = self.scalenet05(alpha**4.0*x)
        y06 = self.scalenet06(alpha**5.0*x)
        y07 = self.scalenet07(alpha**6.0*x)
        y08 = self.scalenet08(alpha**7.0*x)
        y_output1=torch.cat((y01,y02,y03,y04,y05,y06,y07,y08), 1)
        yout = (self.predict(y_output1))                        # linear output
        return yout

def train_u(args, model_list, device, interior_train_loader, 
          dirichlet_bdry_training_data_loader, 
          optimizer2, epoch,lamda,beta): # 还可添加loss_func等参数
    bdryiter= iter(dirichlet_bdry_training_data_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x        =Variable(data,           requires_grad=True )     
        f        =Variable(target[:, 0:2], requires_grad=False)
        divf     =Variable(target[:,   2], requires_grad=False)
        divu_RHS =Variable(target[:,   3], requires_grad=False)

        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)

        loss_u = ResLoss_u(x,bdry_x,f,divu_RHS,bdry_velocity,beta,lamda,model_list,optimizer2,epoch)
        optimizer2.zero_grad()
        loss_u.backward()
        optimizer2.step()
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>5d}  [{:>6d}/{} ({:3.0f}%)]  Loss of u: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_u.item()))
    return loss_u.item()
 
def train_w(args, model_list, device, interior_train_loader, 
          optimizer1, epoch,lamda): # 还可添加loss_func等参数
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x        =Variable(data,           requires_grad=True )     
        f        =Variable(target[:, 0:2], requires_grad=False)
        divf     =Variable(target[:,   2], requires_grad=False)
        divu_RHS =Variable(target[:,   3], requires_grad=False)


        loss_w,loss3 = ResLoss_w(x,f,divu_RHS,model_list,lamda)
        optimizer1.zero_grad()
        loss_w.backward()
        optimizer1.step()
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>5d}  [{:>6d}/{} ({:3.0f}%)]  Difference of grad_u and w: {:.6f}  Loss of w: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss3.item(), loss_w.item()))
    return loss_w.item()

def train_p(args, model_list, device, interior_train_loader, 
          optimizer3, epoch,lamda,beta): # 还可添加loss_func等参数
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x        =Variable(data,           requires_grad=True )     
        f        =Variable(target[:, 0:2], requires_grad=False)
        divf     =Variable(target[:,   2], requires_grad=False)
        divu_RHS =Variable(target[:,   3], requires_grad=False)

        
        loss_p = ResLoss_p(x,f,divf,beta,lamda,model_list)
        optimizer3.zero_grad()
        loss_p.backward()
        optimizer3.step()
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>5d}  [{:>6d}/{} ({:3.0f}%)]  Loss of p: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), loss_p.item()))
    return loss_p.item()

def ResLoss_w(x,f,divu_RHS,model_list,lamda):
    #  the size of x is (batch_size, 2)  
    #  the size of interior_predict is (batch_size, 2)
    #  the size of interior_p_predict is (batch_size, 1)
    #  the size of interior_w_predict is (batch_size, 4)
    interior_u_predict = model_list[0](x) 
    interior_w_pred = model_list[2](x) 
    interior_w_predict = torch.cat((interior_w_pred,interior_w_pred[:,0:1]),1)
    
    # calculate the derivatives:
    # the size of grad is (batch_size, 2) and each row is the (\partial_x u, \partial_y u)
    grad_u1 = torch.autograd.grad(interior_u_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 0])])
    grad_u2 = torch.autograd.grad(interior_u_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 1])])
    
    loss_function = nn.MSELoss()
    
    loss3 = loss_function(grad_u1[0], interior_w_predict[:,0:2])+loss_function(grad_u2[0], interior_w_predict[:, 2:4])
    res = 1*loss3
   
    return res, loss3
    
def ResLoss_u(x,bdry_x,f,divu_RHS,bdry_velocity,beta,lamda,model_list,optimizer,epoch):
        #  the size of x is (batch_size, 2)  
    #  the size of interior_predict is (batch_size, 2)
    #  the size of interior_p_predict is (batch_size, 1)
    #  the size of interior_w_predict is (batch_size, 4)
    interior_u_predict = model_list[0](x) 
    interior_p_predict = model_list[1](x)
    interior_w_pred = model_list[2](x) 
    interior_w_predict = torch.cat((interior_w_pred,interior_w_pred[:,0:1]),1)
 
    bdry_u_predict     = model_list[0](bdry_x)
    # calculate the derivatives:
    # the size of grad is (batch_size, 2) and each row is the (\partial_x u, \partial_y u)
    grad_u1 = torch.autograd.grad(interior_u_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 0])])
    grad_u2 = torch.autograd.grad(interior_u_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 1])])
    grad_p = torch.autograd.grad(interior_p_predict, x,  create_graph=True, grad_outputs=[torch.ones_like(interior_p_predict)])

    u_grad_u1 = torch.sum(interior_u_predict*interior_w_predict[:,0:2], dim=1)
    u_grad_u2 = torch.sum(interior_u_predict*interior_w_predict[:,2:4], dim=1)
    
    grad_u1_2=torch.sum(torch.square(grad_u1[0]),dim=1)
    grad_u2_2=torch.sum(torch.square(grad_u2[0]),dim=1)
    
    loss_function = nn.MSELoss()
    
    loss1 = loss_function(global_nu*grad_u1_2,(-u_grad_u1-grad_p[0][:,0]+f[:,0])*interior_u_predict[:,0])
    loss2 = loss_function(global_nu*grad_u2_2,(-u_grad_u2-grad_p[0][:,1]+f[:,1])*interior_u_predict[:,1])
    loss4 = loss_function(bdry_u_predict, bdry_velocity[:, 0:2])
    res =  loss1 + loss2
    bound = (loss4)              # 调用损失函数计算损失
    
    loss_u = res + lamda * bound
    return loss_u

def ResLoss_p(x,xb,nb,fb,divf, beta,lamda,model_list):
        #  the size of x is (batch_size, 2)  
    #  the size of interior_predict is (batch_size, 2)
    #  the size of interior_p_predict is (batch_size, 1)
    #  the size of interior_w_predict is (batch_size, 4)
    interior_u_predict = model_list[0](x) 
    interior_p_predict = model_list[1](x)

    bdry_u_predict = model_list[0](xb)
    bdry_p_predict = model_list[1](xb)
    bdry_w_pred = model_list[2](xb) 
    bdry_w_predict = torch.cat((bdry_w_pred,bdry_w_pred[:,0:1]),1)
    # calculate the derivatives:
    # the size of grad is (batch_size, 2) and each row is the (\partial_x u, \partial_y u)
    grad_p = torch.autograd.grad(interior_p_predict, x,  create_graph=True, grad_outputs=[torch.ones_like(interior_p_predict)])
    grad_p_2 = torch.sum(torch.square(grad_p),dim=1)
    grad_u1 = torch.autograd.grad(interior_u_predict[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 0])])
    grad_u2 = torch.autograd.grad(interior_u_predict[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict[:, 1])])
 
    grad_p_bdry = torch.autograd.grad(bdry_p_predict, xb,  create_graph=True, grad_outputs=[torch.ones_like(bdry_p_predict)])
    grad_w11= torch.autograd.grad(bdry_w_predict[:, 0], xb,  create_graph=True, grad_outputs=[torch.ones_like(bdry_w_predict[:, 0])])
    grad_w12= torch.autograd.grad(bdry_w_predict[:, 1], xb,  create_graph=True, grad_outputs=[torch.ones_like(bdry_w_predict[:, 1])])
    grad_w21= torch.autograd.grad(bdry_w_predict[:, 2], xb,  create_graph=True, grad_outputs=[torch.ones_like(bdry_w_predict[:, 2])])
    grad_w22= [-grad_w11[0]]
    u_grad_u1 = torch.sum(bdry_u_predict*bdry_w_predict[:, 0:2], dim=1)
    u_grad_u2 = torch.sum(bdry_u_predict*bdry_w_predict[:, 2:4], dim=1)

    divw1=grad_w11[0][:, 0]+grad_w12[0][:, 1]
    divw2=grad_w21[0][:, 0]+grad_w22[0][:, 1]
    p_u_grad_u =  2*(grad_u1[0][:,0]**2+grad_u1[0][:,1]*grad_u2[0][:,0])*interior_p_predict 
#    div_grad_p = interior_w_predict[:,0]**2+interior_w_predict[:,1]*interior_w_predict[:,2]))*interior_p_predict
    p_bound_x =  nb[:,0]*(u_grad_u1-global_nu*divw1+grad_p_bdry[0][:,0])
    p_bound_y = nb[:,1]*(u_grad_u2-global_nu*divw2+grad_p_bdry[0][:,1])
 
   
    loss_function = nn.MSELoss()
    loss1 = loss_function(p_bound_x+p_bound_y,torch.sum(nb*fb,dim=1))
    loss3 = loss_function(-grad_p_2+p_u_grad_u,divf*interior_p_predict)
    bound = loss1 
    res = loss3
    loss_p = beta * res + lamda * bound 
    return loss_p

