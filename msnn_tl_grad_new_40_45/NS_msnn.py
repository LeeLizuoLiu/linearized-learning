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

global_nu=0.05

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
        alpha = 4 
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

def train(args, model_list, device, interior_train_loader, 
          dirichlet_bdry_training_data_loader, coarse_data_loader,
          optimizer, epoch,lamda,beta,gamma): # 还可添加loss_func等参数
    retloss=[]
    bdryiter= iter(dirichlet_bdry_training_data_loader)
    coarseiter = iter(coarse_data_loader)
    for batch_idx, (data, target) in enumerate(interior_train_loader):   # 从数据加载器迭代一个batch的数据
        x        =Variable(data,           requires_grad=True )     
        f        =Variable(target[:, 0:2], requires_grad=False)
        divf     =Variable(target[:,   2], requires_grad=False)
        divu_RHS =Variable(target[:,   3], requires_grad=False)

        bdrydata, bdrytarget=bdryiter.next()
        bdry_x       =Variable(  bdrydata, requires_grad=False)
        bdry_velocity=Variable(bdrytarget, requires_grad=False)
        loss_total,res,bound = ResLoss_upw(x,bdry_x,f,divu_RHS,divf,bdry_velocity,beta,lamda,model_list,epoch)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        lamda_temp = lamda

        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {:>5d}  [{:>6d}/{} ({:3.0f}%)] Loss of res: {:.6f} Loss of bound: {:.6f}'.format(
                epoch, batch_idx * len(data), len(interior_train_loader.dataset),
                100. * batch_idx / len(interior_train_loader), res.item(),bound.item()))
        if batch_idx==0:
            retloss=loss_total
            retbound = 0
            retres = 0
            retcoar_loss = 0

    return retloss, lamda_temp, retbound, retres, retcoar_loss
def ResLoss_upw(x,bdry_x,f,divu_RHS,divf,bdry_velocity,beta,lamda,model_list,epoch):
    
    interior_u_predict_old = model_list[0](x) 
    interior_u_predict_new = model_list[1](x) 
    interior_p_predict = model_list[2](x)
    bdry_u_predict     = model_list[1](bdry_x)
 
    # calculate the derivatives:
    # the size of grad is (batch_size, 2) and each row is the (\partial_x u, \partial_y u)
    grad_u1_old = torch.autograd.grad(interior_u_predict_old[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict_old[:, 0])])
    grad_u2_old = torch.autograd.grad(interior_u_predict_old[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict_old[:, 1])])
    grad_u1_new = torch.autograd.grad(interior_u_predict_new[:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict_new[:, 0])])
    grad_u2_new = torch.autograd.grad(interior_u_predict_new[:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(interior_u_predict_new[:, 1])])
    
    grad_w11= torch.autograd.grad(grad_u1_new[0][:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_u1_new[0][:, 0])])
    grad_w12= torch.autograd.grad(grad_u1_new[0][:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_u1_new[0][:, 1])])
    grad_w21= torch.autograd.grad(grad_u2_new[0][:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_u1_new[0][:, 0])])
    grad_w22= torch.autograd.grad(grad_u2_new[0][:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_u1_new[0][:, 1])])

    grad_p = torch.autograd.grad(interior_p_predict, x,  create_graph=True, grad_outputs=[torch.ones_like(interior_p_predict)])
    pxx = torch.autograd.grad(grad_p[0][:, 0], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_p[0][:, 0])])
    pyy = torch.autograd.grad(grad_p[0][:, 1], x,  create_graph=True, grad_outputs=[torch.ones_like(grad_p[0][:, 1])])


    u_grad_u1 = torch.sum(interior_u_predict_old*grad_u1_new[0], dim=1)
    u_grad_u2 = torch.sum(interior_u_predict_old*grad_u2_new[0], dim=1)

    divu = grad_u1_new[0][:,0]+grad_u2_new[0][:,1]
 
    divw1=grad_w11[0][:, 0]+grad_w12[0][:, 1]
    divw2=grad_w21[0][:, 0]+grad_w22[0][:, 1]

    div_grad_p =  pxx[0][:, 0]+pyy[0][:, 1]+2*(-grad_u2_new[0][:,1]*grad_u1_new[0][:,0]+grad_u1_new[0][:,1]*grad_u2_new[0][:,0])
 
    loss_function = nn.MSELoss()
    
    loss1 = loss_function(beta*u_grad_u1-global_nu*divw1+grad_p[0][:,0], f[:,0])
    loss2 = loss_function(beta*u_grad_u2-global_nu*divw2+grad_p[0][:,1], f[:,1])
    loss4 = loss_function(bdry_u_predict, bdry_velocity[:, 0:2])
    loss5 = loss_function(divu, divu_RHS)
    loss6 = loss_function(div_grad_p,divf)
  
    res = (loss1 + loss2)  + loss5 + loss6
    bound = (loss4)              # 调用损失函数计算损失
    loss = beta * res + lamda * bound 
    
    return loss,res,bound

