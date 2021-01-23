# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:22:26 2020

generate data for solving Poisson equation

    -\Delta u=f   in \Omega=[0,1]^2
    
with exact solution 
     
     u=sin(\pi x)sin(\pi y)     f=2\pi^2u
     
@author: Bo Wang
"""
import random
import numpy as np
import matplotlib.pyplot as plt


nu=0.05
Re=1.0/nu
lambda_const=Re/2.0-np.sqrt(Re*Re/4.0+4.0*np.pi*np.pi)
n_freq=50
m_freq=55
def target(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    cosx1x2=np.reshape(np.cos(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    sinx1x2=np.reshape(np.sin(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    f1= expx1*(lambda_const*expx1+(lambda_const-4*(n_freq*n_freq+m_freq*m_freq-1)*nu*np.pi*np.pi)*cosx1x2-4*nu*n_freq*np.pi*lambda_const*sinx1x2)
    f2=n_freq/m_freq*expx1*(4*(n_freq*n_freq+m_freq*m_freq-3)*nu*np.pi*np.pi-3*lambda_const)*cosx1x2+lambda_const*expx1/(2*m_freq*np.pi)*(4*(3*n_freq*n_freq+m_freq*m_freq-1)*nu*np.pi*np.pi-lambda_const)*sinx1x2
    return np.concatenate((f1, f2), axis=1)

def div_f(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    divf=2.0*lambda_const*lambda_const*expx1*expx1
    return divf

def velocity(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    cosx1x2=np.reshape(np.cos(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    sinx1x2=np.reshape(np.sin(2.0*n_freq*np.pi*x[:, 0]+2.0*m_freq*np.pi*x[:, 1]), (x.shape[0], 1))
    yb1=1.0-expx1*cosx1x2
    yb2=lambda_const*expx1*sinx1x2/(2.0*m_freq*np.pi)+n_freq/m_freq*expx1*cosx1x2
    return np.concatenate((yb1, yb2), axis=1)

def pressure(x):
    expx1=np.reshape(np.exp(lambda_const*x[:, 0]), (x.shape[0], 1))
    return expx1*expx1/2.0 


def generate_data():
    centers=np.array([[0.5, 1.25, 1.3, 0.5, 1.2, 1.6], [0.0, -0.2, 0.4, 1.1, 0.9, 1]]).T
    
    radius =np.array([0.2, 0.15, 0.18, 0.2, 0.18, 0.15]);
    
    n=3200*10*20
    x = np.array([[random.uniform(0,2) for _ in range(n)],[random.uniform(-0.5, 1.5) for _ in range(n)]]).T
    for i in range(6):
        x=x[(x[:,0]-centers[i][0])**2+(x[:,1]-centers[i][1])**2>radius[i]**2]
    x = x[0:3200*20,:]
        
#     plt.scatter(x[:, 0], x[:, 1])
    f= target(x)
    divf=div_f(x)
    
    
    np.set_printoptions(threshold=np.inf)
    
    # boundary values of velocity
    nb=6400
    xb1=np.array([[random.uniform(0,2) for _ in range(nb)],[-0.5]*nb]).T
    xb2=np.array([[random.uniform(0, 2) for _ in range(nb)],[1.5]*nb]).T
    xb3=np.array([[0.0]*nb,[random.uniform(-0.5,1.5) for _ in range(nb)]]).T
    xb4=np.array([[2.0]*nb, [random.uniform(-0.5,1.5) for _ in range(nb)]]).T
    xb=np.concatenate((xb1, xb2, xb3, xb4), axis=0)
    
    ncirc=nb
    for i in range(6):
        theta=np.array([random.uniform(0,2*np.pi) for _ in range(ncirc)])
        xbcirc=np.array([radius[i]*np.cos(theta)+centers[i][0], radius[i]*np.sin(theta)+centers[i][1]]).T
        xb=np.concatenate((xb, xbcirc), axis=0)
    
#     plt.scatter(xb[:, 0], xb[:, 1])
#    print(xb)
#    print(np.exp(lambda_const*xb[:, 0]))
    yb=velocity(xb)
    
    return x, np.concatenate((f, divf), axis=1),xb,yb