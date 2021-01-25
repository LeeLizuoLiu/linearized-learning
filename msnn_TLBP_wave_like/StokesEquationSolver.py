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
from utils.utils import FullyConnectedNet,load_mesh,load_pretrained_model,evaluate,plot_contourf_vorticity,DatasetFromTxt,generate_data,StatGrad,data_generator
from NS_msnn import MultiScaleNet,train,global_nu



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
    parser.add_argument('--seed', type=int, default=1213, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    filepath =os.path.abspath(os.path.join(os.getcwd(),os.pardir))      
    use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU
    print('use cuda:', use_cuda)    
    torch.manual_seed(args.seed) # 设置随机种子，保证可重复性
    
    device = torch.device("cuda:0" if use_cuda else "cpu") # 设置使用CPU or GPU
    
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    
    nNeuron=100
    nb_head = 18
    
    model_u = MultiScaleNet(2, 2, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_w = MultiScaleNet(2, 3, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    model_p = MultiScaleNet(2, 1, hidden_size= nNeuron,nb_heads=nb_head).to(device)	# 实例化自定义网络模型
    
    loadepochs=0
    
    if loadepochs!=0:
        load_pretrained_model(model_u, 'netsave/u_net_params_at_epochs'+str(loadepochs)+'.pkl')
        load_pretrained_model(model_w, 'netsave/w_net_params_at_epochs'+str(loadepochs)+'.pkl')
        load_pretrained_model(model_p, 'netsave/p_net_params_at_epochs'+str(loadepochs)+'.pkl')
        
    len_inte_data = 3200*args.nbatch 
    len_bound_data = 320*args.nbatch
    
    paramsw = list(model_w.parameters())
    paramsu = list(model_u.parameters())
    paramsp = list(model_p.parameters())
    optimizer = optim.Adam(paramsw+paramsu+paramsp, lr=args.lr) # 实例化求解器

    model_list=[model_u,  model_p, model_w]  
    CollectNorm = {}
    for name, _ in model_u.named_children():
        CollectNorm[name] = []

    coarse_training_dataset = DatasetFromTxt(filepath+'/Data/uvp.txt',device)    
    coarse_data_loader = torch.utils.data.DataLoader(coarse_training_dataset,
                                                    batch_size=int(len(coarse_training_dataset)/args.nbatch),
                                                    shuffle=False,
                                                    **kwargs)    
    points, elements = load_mesh(filepath+"/Data/mesh_file.dat")
    XY=Variable(torch.tensor(points),requires_grad=True).to(device)
    
    loss=np.array([0.0]*(args.epochs-loadepochs))
    lamda = 10000. 
    beta = 1 
    gamma = 0.
    res_temp = 1e12
    bound_temp = 1e12
    coarse_loss = 0
    train_data = data_generator(10,15,global_nu)
    for epoch in range(loadepochs+1, args.epochs + 1): # 循环调用train() and test()进行epoch迭代
        if  coarse_loss< 3000 and epoch%5==1 :
            x_int, inter_target, xlxb, ulub  = generate_data(len_inte_data,len_bound_data)
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

        loss[epoch-loadepochs-1],lamda_temp, bound, res, coarse_loss=train(args, model_list, device, interior_training_data_loader, 
                                                   dirichlet_boundary_training_data_loader, 
                                                   coarse_data_loader,
                                                   optimizer, epoch, lamda,beta,gamma,CollectNorm,args.epochs)

        if epoch%20==0:
            torch.save(model_u.state_dict(), 'netsave/u_net_params_at_epochs'+str(epoch)+'.pkl') 
            torch.save(model_p.state_dict(), 'netsave/p_net_params_at_epochs'+str(epoch)+'.pkl') 
            torch.save(model_w.state_dict(), 'netsave/w_net_params_at_epochs'+str(epoch)+'.pkl')  
            plot_contourf_vorticity(model_w,model_u, epoch,points, elements,XY)
        lamda = lamda_temp
        if epoch%1000000==0:
                gamma = gamma/1.03
#            if bound<= bound_temp* 0.95:
#                beta = beta*1.
#                bound_temp = bound 
#            else:
#                lamda = lamda/1.0
 
    import json

    with open('CollectNorm.json', 'w') as fp:
        json.dump(CollectNorm, fp)

    plt.plot(np.array(range(loadepochs, args.epochs)), loss, 'b-', lw=2)
    plt.yscale('log')
    plt.savefig('result_plots/loss'+str(loadepochs)+'to'+str(args.epochs)+'.pdf')
    plt.close()

    for key in CollectNorm.keys():
        plt.plot(np.array(range(1,1+len(CollectNorm[key]))),np.array(CollectNorm[key]),label=key)
    plt.legend(loc = 'lower right')
    plt.savefig('result_plots/GradientsStat.pdf')
    evaluate(model_u,device,epoch)
