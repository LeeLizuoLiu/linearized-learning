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
import torch.optim as optim	                  # 实现各种优化算法的包
import pdb
import os
import sys
sys.path.append("..")
from utils.utils import optimWithExpDecaylr, plot_sol,FullyConnectedNet,load_mesh,load_pretrained_model,evaluate,plot_u,DatasetFromTxt,generate_data_dirichlet,StatGrad,data_generator
from NS_msnn import MultiScaleNet,train,global_nu
import logging


if __name__ == '__main__':
   # Training settings
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--nbatch', type=int, default=50, metavar='N',
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
    parser.add_argument('--seed', type=int, default=12198789792343, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr-adjust-step', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--check-step', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')

   
    args = parser.parse_args()
    filepath =os.path.abspath(os.path.join(os.getcwd(),os.pardir))      
    use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU
    print('use cuda:', use_cuda)    
    torch.manual_seed(args.seed) # 设置随机种子，保证可重复性
    
    device = torch.device("cuda:0" if use_cuda else "cpu") # 设置使用CPU or GPU
    
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    
    nNeuron=128
    nb_head = 18
    
    model_u_old =MultiScaleNet(2, 2, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_p = FullyConnectedNet(2,nNeuron,1).to(device)	# 实例化自定义网络模型
    model_u_new = MultiScaleNet(2, 2, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    
    loadepochs=0

    if loadepochs!=0:
        load_pretrained_model(model_u_old, 'netsave/old_u_net_params_at_epochs'+str(loadepochs)+'.pkl')
        load_pretrained_model(model_p, 'netsave/p_net_params_at_epochs'+str(loadepochs)+'.pkl')
        load_pretrained_model(model_u_new, 'netsave/old_u_net_params_at_epochs'+str(loadepochs)+'.pkl')
        
    len_inte_data = 2048*args.nbatch 
    len_bound_data = 256*args.nbatch
    
    paramsw = list(model_u_new.parameters())
    paramsp = list(model_p.parameters())
    optimizer = optim.Adam(paramsw+paramsp, lr=args.lr) # 实例化求解器

    model_list=[model_u_old,model_u_new,model_p]  
    
    coarse_training_dataset = DatasetFromTxt(filepath+'/Data/uvp.txt',device)    
    coarse_data_loader = torch.utils.data.DataLoader(coarse_training_dataset,
                                                    batch_size=int(len(coarse_training_dataset)/args.nbatch),
                                                    shuffle=False,
                                                    **kwargs)    
    points, elements = load_mesh(filepath+"/Data/mesh_file.dat")
    XY=Variable(torch.tensor(points),requires_grad=True).to(device)
    
    loss=np.array([0.0]*(args.epochs-loadepochs))
    lamda = 1e4 
    beta = 1.
    gamma = 0.
    res_temp = 1e12
    bound_temp = 1e12
    coarse_loss = 0
    Loss_reshold =  1e12
    lr_adjust_step = args.lr_adjust_step
    check_step = args.check_step
    lr = args.lr
    delta_lr = args.lr/(args.epochs/lr_adjust_step)
    train_data = data_generator(30,35,global_nu)
    plot_sol_drawer = plot_sol(30,35,global_nu)
    logging.basicConfig(filename='lrstep'+str(lr_adjust_step)+'check'+str(check_step)+'.log', 
                        filemode='w',
                        format='%(asctime)s - %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO) 
    for epoch in range(loadepochs+1, args.epochs+1): # 循环调用train() and test()进行epoch迭代
        if  coarse_loss< 3000 and epoch%1==0 :
            x_int, inter_target, xlxb, ulub  = train_data.generate(len_inte_data,len_bound_data)
            interior_training_dataset=torch.utils.data.TensorDataset(torch.Tensor(x_int).to(device),torch.Tensor(inter_target).to(device))
            interior_training_data_loader = torch.utils.data.DataLoader(interior_training_dataset, 
                                                                        batch_size=int(len_inte_data/args.nbatch),
                                                                        shuffle=True, 
                                                                        **kwargs)

            dirichlet_boundary_training_dataset= torch.utils.data.TensorDataset(torch.Tensor(xlxb).to(device),torch.Tensor(ulub).to(device))
            dirichlet_boundary_training_data_loader =torch.utils.data.DataLoader(dirichlet_boundary_training_dataset,
                                                                                batch_size=int(len(dirichlet_boundary_training_dataset)/args.nbatch), 
                                                                                shuffle=True, 
                                                                                **kwargs)

        loss[epoch-1-loadepochs],lamda_temp, bound, res, coarse_loss=train(args, model_list, device, interior_training_data_loader, 
                                                   dirichlet_boundary_training_data_loader, 
                                                   coarse_data_loader,
                                                   optimizer, epoch, lamda,beta,gamma)

        if epoch%(check_step)==0 and loss[epoch-1-loadepochs]<0.9*Loss_reshold:
            torch.save(model_u_new.state_dict(), 'netsave/old_u_net_params_at_epochs'+str(epoch)+'.pkl') 
            load_pretrained_model(model_u_old, 'netsave/old_u_net_params_at_epochs'+str(epoch)+'.pkl')
            Loss_reshold = loss[epoch-1-loadepochs]
            torch.save(model_p.state_dict(), 'netsave/p_net_params_at_epochs'+str(epoch)+'.pkl') 
 
        if epoch%50==0:
            plot_sol_drawer.plot_pressure_along_line(model_p,epoch,device)
            plot_sol_drawer.plot_velocity_along_line(model_u_new,epoch,device)
            torch.save(model_u_new.state_dict(), 'netsave/old_u_net_params_at_epochs'+str(epoch)+'.pkl') 
            torch.save(model_p.state_dict(), 'netsave/p_net_params_at_epochs'+str(epoch)+'.pkl') 

        if epoch%lr_adjust_step==1:
            optimizer = optim.Adam(paramsw+paramsp, lr=lr) # 实例化求解器
            lr = lr - delta_lr
           

        lamda = lamda_temp

    plt.plot(np.array(range(loadepochs, args.epochs)), loss, 'b-', lw=2)
    plt.yscale('log')
    plt.savefig('result_plots/loss'+str(loadepochs)+'to'+str(args.epochs)+'.pdf')
    plt.close()

    np.save('loss',loss)