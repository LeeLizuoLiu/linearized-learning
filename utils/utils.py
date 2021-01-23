import sys
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import os
import random

def normalizeGrad(model):
    for child in model.children():
        grads = 0 
        for param in child.parameters():
            grads  = grads + torch.sum(torch.square(param.grad))
        for param in child.parameters():
            param.grad = param.grad/torch.sqrt(grads)

def adjustGrad(model):
    for child in model.children():
        grads = 0 
        for param in child.parameters():
            grads  = grads + torch.sum(torch.square(param.grad))
        if torch.sqrt(grads).item()<100:
            for param in child.parameters():
                param.grad = param.grad*0

def StatGrad(model,collectNorm): 
    for name, child in model.named_children():
        grads = 0 
        for param in child.parameters():
            grads  = grads + torch.sum(torch.square(param.grad))
        collectNorm[name].append(grads.item())
    return collectNorm



def gradient(params):
    grads = []
    for param in params:
        if param.grad ==None:
            continue
        else:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)   
    return grads

def save_gradient(params,name):
    grads = []
    for param in params:
        if param.grad ==None:
            continue
        else:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)   
    torch.save(grads, 'gradients/'+name+'.pt')

def Compute_gradient(optimizer, params, loss,name):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for param in params:
        if param.grad ==None:
            continue
        else:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)   
    torch.save(grads, 'gradients/'+name+'.pt')

def compute_lamda(res, bound):
    lamda  = torch.mean(torch.abs(res))/torch.mean(torch.abs(bound))
    return lamda

def evaluate(model_u,device,epoch):
    filepath =os.path.abspath(os.path.join(os.getcwd(),os.pardir))
    if os.path.isfile(filepath+'/Data/velocity_x@__57.txt'):
       yu = np.loadtxt(filepath+'/Data/velocity_x@__57.txt').astype(np.float32)
       u = yu[:,1]
       x = np.ones_like(yu)*-.57
       x[:,1] = yu[:,0]
       predict = model_u(torch.Tensor(x).to(device)).cpu().data.numpy()
       err = abs(predict[:,0]-u)
       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       tcf = ax.plot(yu[:,0],err)
       fig.savefig('result_plots/err_u'+str(epoch)+'.pdf')
       plt.close()

       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       ax.plot(yu[:,0],u)
       ax.plot(yu[:,0],predict[:,0])
       fig.savefig('result_plots/u'+str(epoch)+'.pdf')
       plt.close()

def check_layers(model):
    i = 0
    for child in model.children():
        print("i:",i)
        i = i+1
        j = 0
        for param in child.parameters():
            print("j:", j)
            j = j+1

def freeze(model):
    print("Freezing the learned subnetworks ...")
    cntr = 1
    total = 20 
    for child in model.children():
        if cntr < total:
            print("Freezed subnetwork No.:{}".format(cntr))
            for param in child.parameters():
                param.requires_grad = False
        cntr +=1

def defreeze(model):
    print("Defreezing the learned subnetworks ...")
    cntr = 1
    total = 20 
    for child in model.children():
        if cntr < total:
            print("Subnetwork No.:{} is defrozen".format(cntr))
            for param in child.parameters():
                param.requires_grad = True
        cntr +=1


def load_pretrained_model(NewModel,SavedModelLocation):
    print("Loading learned subnetworks ...")
    pretrained_dict = torch.load(SavedModelLocation)
    NewModelDict = NewModel.state_dict()
    for name, param in NewModelDict.items():
        if name not in pretrained_dict:
            continue
        NewModelDict[name].copy_(pretrained_dict[name])

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

def plot_heatmap_v(model_w,model_u, epoch, points,elements,XY):
    x, y = points.T
    triangulation = tri.Triangulation(x, y, elements)
    
    velocity = model_u(XY)

    plt.figure()
    plt.tricontourf(triangulation, velocity[:,0].cpu().data.numpy(), cmap='rainbow')
    plt.axis('equal')
    plt.savefig('result_plots/Epoch_'+str(epoch)+'velocity_contour.pdf')
    plt.close()

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

class phi(nn.Module):
    def __init__(self):
        super(phi,self).__init__()
    def forward(self,x):
        return torch.sin(x)

class DatasetFromTxt(torch.utils.data.Dataset):
    def __init__(self, filepath,device, input_transform=None, target_transform=None):
        super(DatasetFromTxt, self).__init__()

        self.data_in_file = torch.Tensor(np.loadtxt(filepath).astype(np.float32)).to(device)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = self.data_in_file[index][0:2]
        
        target = self.data_in_file[index][2:]
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.data_in_file)

def f(x):
    z=np.zeros_like(x)
    return z

def div_f(x):
    return np.zeros((len(x), 1))

def uT(x):
    u = (np.cos(10*np.pi*x[:,1])+1)*(np.cos(8*np.pi*x[:,0])+1)
    ux = -8*np.pi*np.sin(8*np.pi*x[:,0])*(np.cos(10*np.pi*x[:,1])+1)
    uy = -10*np.pi*np.sin(10*np.pi*x[:,1])*(np.cos(8*np.pi*x[:,0])+1)
    uyy = -100*np.pi**2*np.cos(10*np.pi*x[:,1])*(np.cos(8*np.pi*x[:,0])+1)
    uxx = -64*np.pi**2*np.cos(8*np.pi*x[:,0])*(np.cos(10*np.pi*x[:,1])+1)
    return u, ux, uy, uxx, uyy
def vT(x):
    v = (np.sin(np.pi*x[:,1])+1)*(np.sin(3*np.pi*x[:,0])+1)
    vx = 3*np.pi*np.cos(3*np.pi*x[:,0])*(np.sin(np.pi*x[:,1])+1)
    vy = np.pi*np.cos(np.pi*x[:,1])*(np.sin(3*np.pi*x[:,0])+1)
    vyy = -np.pi**2*np.sin(np.pi*x[:,1])*(np.sin(3*np.pi*x[:,0])+1)
    vxx = -9*np.pi**2*np.sin(3*np.pi*x[:,0])*(np.sin(np.pi*x[:,1])+1)
    return v,vx,vy,vxx,vyy

def velocity(x):
    nu=0.1
    Re=1.0/nu
    lambda_const=Re/2.0-np.sqrt(Re*Re/4.0+4.0*np.pi*np.pi)
    n_freq=30
    m_freq=35
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    cosx1x2=np.reshape(np.cos(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    sinx1x2=np.reshape(np.sin(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    yb1=1.0-expx1*cosx1x2
    yb2=lambda_const*expx1*sinx1x2/(2.0*m_freq*np.pi)+n_freq/m_freq*expx1*cosx1x2
    return np.concatenate((yb1, yb2), axis=1)
 
def generate_data_dirichlet(int_datasize,nb):
    n=10*int_datasize
    x = np.array([[random.uniform(0,6) for _ in range(n)],[random.uniform(0, 3) for _ in range(n)]]).T
    
    centers=np.array([1.5, 1.5])
    x=x[(x[:,0]-centers[0])**2+(x[:,1]-centers[1])**2>0.04]
    x_int = x[0:int_datasize]    
    
    fdata= f(x_int)
    divf=div_f(x_int)    
    divu_RHS = div_f(x_int)
    inter_target=np.concatenate((fdata, divf, divu_RHS), axis=1)
    
    # boundary values of velocity
    xb1=np.array([[random.uniform(0,6) for _ in range(nb)],[0]*nb]).T
    xb2=np.array([[random.uniform(0,6) for _ in range(nb)],[3]*nb]).T

    xb3=np.array([[0]*nb, [random.uniform(0,3) for _ in range(nb)]]).T
    xb4=np.array([[6]*nb, [random.uniform(0,3) for _ in range(nb)]]).T
        
    theta=np.array([random.uniform(0,2.0*np.pi) for _ in range(nb)])
    r = 0.2
    xbcirc=np.array([r*np.cos(theta), r*np.sin(theta)]).T
        
    xb_dirichlet=np.concatenate((xb1, xb2,xb3,xb4, xbcirc), axis=0)
    vel = velocity(xb_dirichlet)

    return x_int, inter_target, xb_dirichlet,vel

def genData_test(int_datasize,nb):
    n=int_datasize
    x = np.array([[random.uniform(-2.5,3.5) for _ in range(n)],[random.uniform(-1.5, 1.5) for _ in range(n)]]).T
    x_int = x[0:int_datasize]    

    nu = 0.01
    fdata= f(x_int)
    u,ux,uy,uxx,uyy = uT(x_int)
    v,vx,vy,vxx,vyy = vT(x_int)
    fdata[:,0] = -nu*uxx-nu*uyy + (u*ux+v*uy)
    fdata[:,1] = -nu*vxx-nu*vyy + (u*vx+v*vy)

    divf=div_f(x_int)    
    divu_RHS = div_f(x_int)
    inter_target=np.concatenate((fdata, divf, divu_RHS), axis=1)
    
    # boundary values of velocity
    xb1=np.array([[random.uniform(-2.5,3.5) for _ in range(nb)],[-1.5]*nb]).T
    xb2=np.array([[random.uniform(-2.5,3.5) for _ in range(nb)],[ 1.5]*nb]).T
    xb3=np.array([[-2.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
    xb4=np.array([[ 3.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
    xb_dirichlet=np.concatenate((xb1, xb2,xb3,xb4), axis=0)

    ul = np.zeros_like(xb_dirichlet)
    ub,_,_,_,_ = uT(xb_dirichlet)
    vb,_,_,_,_ = vT(xb_dirichlet)
    ul[:,0] = ub
    ul[:,1] = vb
    
    xr_dirichlet = xb4
    pr = np.zeros((len(xr_dirichlet),1))

    return x_int, inter_target, xb_dirichlet, ul, xr_dirichlet, pr

def genData_bhd(int_datasize,nb):
    n=10*int_datasize
    x = np.array([[random.uniform(0.5,3.5) for _ in range(n)],[random.uniform(-1.5, 1.5) for _ in range(n)]]).T
    
    centers=np.array([0.0, 0.0])
    x=x[(x[:,0]-centers[0])**2+(x[:,1]-centers[1])**2>0.25]
    x_int = x[0:int_datasize]    
    
    fdata= f(x_int)
    divf=div_f(x_int)    
    divu_RHS = div_f(x_int)
    inter_target=np.concatenate((fdata, divf, divu_RHS), axis=1)
    
    # boundary values of velocity
    xb1=np.array([[random.uniform(0.5,3.5) for _ in range(nb)],[-1.5]*nb]).T
    xb2=np.array([[random.uniform(0.5,3.5) for _ in range(nb)],[ 1.5]*nb]).T

    xb3=np.array([[-2.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
    xb4=np.array([[ 3.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
        
    theta=np.array([random.uniform(0,2.0*np.pi) for _ in range(nb)])
    r = 0.5
    xbcirc=np.array([r*np.cos(theta), r*np.sin(theta)]).T
        
    xb_dirichlet=np.concatenate((xb1, xb2, xbcirc), axis=0)

    xl_dirichlet = xb3
    ul = np.zeros_like(xl_dirichlet)
    ul[:,0] =  10*(np.cos(10*np.pi*xl_dirichlet[:,1])+1)

    ub=np.zeros_like(xb_dirichlet)
    xlxb=np.concatenate((xl_dirichlet,xb_dirichlet),axis=0)
    ulub=np.concatenate((ul, ub), axis=0)
    
    xr_dirichlet = xb4
    pr = np.zeros((len(xr_dirichlet),1))

    return x_int, inter_target, xlxb, ulub, xr_dirichlet, pr

def generate_data(int_datasize,nb):
    n=10*int_datasize
    x = np.array([[random.uniform(-2.5,3.5) for _ in range(n)],[random.uniform(-1.5, 1.5) for _ in range(n)]]).T
    
    centers=np.array([0.0, 0.0])
    x=x[(x[:,0]-centers[0])**2+(x[:,1]-centers[1])**2>0.25]
    x_int = x[0:int_datasize]    
    
    fdata= f(x_int)
    divf=div_f(x_int)    
    divu_RHS = div_f(x_int)
    inter_target=np.concatenate((fdata, divf, divu_RHS), axis=1)
    
    # boundary values of velocity
    xb1=np.array([[random.uniform(-2.5,3.5) for _ in range(nb)],[-1.5]*nb]).T
    xb2=np.array([[random.uniform(-2.5,3.5) for _ in range(nb)],[ 1.5]*nb]).T

    xb3=np.array([[-2.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
    xb4=np.array([[ 3.5]*nb, [random.uniform(-1.5,1.5) for _ in range(nb)]]).T
        
    theta=np.array([random.uniform(0,2.0*np.pi) for _ in range(nb)])
    r = 0.5
    xbcirc=np.array([r*np.cos(theta), r*np.sin(theta)]).T
        
    xb_dirichlet=np.concatenate((xb1, xb2, xbcirc), axis=0)

    xl_dirichlet = xb3
    ul = np.zeros_like(xl_dirichlet)
    ul[:,0] = 10*(np.cos(10*np.pi*xl_dirichlet[:,1])+1)

    ub=np.zeros_like(xb_dirichlet)
    xlxb=np.concatenate((xl_dirichlet,xb_dirichlet),axis=0)
    ulub=np.concatenate((ul, ub), axis=0)
    
    xr_dirichlet = xb4
    pr = np.zeros((len(xr_dirichlet),1))

    return x_int, inter_target, xlxb, ulub, xr_dirichlet, pr 

class MultiScaleNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size = 32, nb_heads=1):
        super(MultiScaleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_dim * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
            phi(),
            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
            phi(),
            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
            phi(),
            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=hidden_size * nb_heads, kernel_size=1, groups=nb_heads),
            phi(),
            nn.Conv1d(in_channels=hidden_size * nb_heads, out_channels=1 * nb_heads, kernel_size=1, groups=nb_heads),
            nn.Conv1d(in_channels=1 * nb_heads, out_channels=output_dim, kernel_size=1)
        )
        self.nb_heads = nb_heads
        self.register_buffer('scale',torch.Tensor([[1.5**(j) for i in range(input_dim)] for j in range(nb_heads)]).view(input_dim*nb_heads,1))
      
    def forward(self, x):
        x = x.repeat(1, self.nb_heads).unsqueeze(-1)
        x = x*self.scale
        flat = self.network(x)
        return torch.squeeze(flat)
