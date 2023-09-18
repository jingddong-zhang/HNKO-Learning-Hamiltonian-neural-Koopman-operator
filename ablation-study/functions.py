import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import cm
import matplotlib as mpl
from torchdiffeq import odeint
import geotorch
torch.set_default_dtype(torch.float64)

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 236/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
]

def t_energy(X):  #total energy
    x,y,a,b = X[:,0],X[:,1],X[:,2],X[:,3]
    h = (a**2+b**2)/2-1/(np.sqrt(x**2+y**2))
    return h

def coord_MSE(X,Y): # X:true, Y:pred, [:,2]
    err = np.sum((X-Y)[:,:2]**2,axis=1)
    return err
def k_energy(X):  #kinetic energy
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = (a**2+b**2)/2
    return h
def p_energy(X):  #potential energy
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = -1/(np.sqrt(x**2+y**2))
    return h
def angular(X):  #Angular Momentum
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = x*b-y*a
    return h

def plot_8():  # X:true, Y:pred
    fontsize = 15
    labelsize = 13
    X = np.load('./data/true_0.03.npy')
    Y = np.load('./data/noko_0.03.npy')
    Z = np.load('./data/hnn_0.03.npy')
    W = np.load('./data/edmd_0.03.npy')
    plt.subplot(241)
    plt.plot(X[:,0],X[:,1],label='True',color='black')
    plt.plot(Y[:,0],Y[:,1],label='HNKO',color=colors[2])
    plt.ylabel('Trajectory',fontsize=fontsize)
    # plt.ylabel('y',fontsize=fontsize)
    plt.title('HNKO',fontsize=fontsize)
    plt.xlabel(r'$x$',fontsize=fontsize)
    plt.xticks([-1,0,1],fontsize = labelsize)
    plt.yticks([-1,0,1],fontsize = labelsize)
    plt.legend(frameon=False)
    plt.subplot(242)
    plt.plot(X[:,0],X[:,1],label='True',color='black')
    plt.plot(Z[:,0],Z[:,1],label='HNN',color=colors[1])
    plt.title('HNN',fontsize=fontsize)
    plt.xticks([-1,0,1],fontsize = labelsize)
    plt.yticks([-1,0,1],fontsize = labelsize)
    plt.legend(frameon=False)
    plt.subplot(243)
    plt.plot(X[:,0],X[:,1],label='True',color='black')
    plt.plot(W[:,0],W[:,1],label='EDMD',color=colors[0])
    plt.title('EDMD',fontsize=fontsize)
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    plt.legend(frameon=False)
    plt.subplot(244)
    plt.plot(np.arange(len(coord_MSE(X,Y))),coord_MSE(X,Y),color=colors[2],label='HNKO')
    plt.plot(np.arange(len(coord_MSE(X,Z))),coord_MSE(X,Z),color=colors[1],label='HNN')
    plt.plot(np.arange(len(coord_MSE(X,W))),coord_MSE(X,W),color=colors[0],label='EDMD')
    plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    plt.yticks([0,0.05,0.1],fontsize = labelsize)
    plt.title('MSE(Coordinates)',fontsize=fontsize)
    plt.legend(frameon=False)
    plt.subplot(245)
    plt.plot(np.arange(len(k_energy(Y))),k_energy(Y),label='Kinetic',color=colors[2],ls='--')
    plt.plot(np.arange(len(p_energy(Y))),p_energy(Y),label='Total',color=colors[2],ls='dotted')
    plt.plot(np.arange(len(t_energy(Y))),t_energy(Y),label='Potential',color=colors[2],ls='-')
    plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    plt.ylabel('Energy',fontsize=fontsize)
    plt.xlabel('Time',fontsize=fontsize)
    plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    plt.legend(frameon=False,loc=5)
    plt.subplot(246)
    plt.plot(np.arange(len(k_energy(Z))),k_energy(Z),label='Kinetic',color=colors[1],ls='--')
    plt.plot(np.arange(len(p_energy(Z))),p_energy(Z),label='Total',color=colors[1],ls='dotted')
    plt.plot(np.arange(len(t_energy(Z))),t_energy(Z),label='Potential',color=colors[1],ls='-')
    plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    plt.legend(frameon=False,loc=5)
    plt.subplot(247)
    plt.plot(np.arange(len(k_energy(W))),k_energy(W),label='Kinetic',color=colors[0],ls='--')
    plt.plot(np.arange(len(p_energy(W))),p_energy(W),label='Total',color=colors[0],ls='dotted')
    plt.plot(np.arange(len(t_energy(W))),t_energy(W),label='Potential',color=colors[0],ls='-')
    plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    plt.legend(frameon=False,loc=5)
    plt.subplot(248)
    plt.plot(np.arange(len(t_energy(X))),np.sqrt((t_energy(X)-t_energy(Y))**2),color=colors[2],label='HNKO')
    plt.plot(np.arange(len(t_energy(Y))),np.sqrt((t_energy(X)-t_energy(Z))**2),color=colors[1],label='HNN')
    plt.plot(np.arange(len(t_energy(Z))),np.sqrt((t_energy(X)-t_energy(W))**2),color=colors[0],label='EDMD')
    plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    plt.yticks([0,0.1,0.2],fontsize = labelsize)
    plt.title('MSE(Total Energy)',fontsize=fontsize)
    plt.legend(frameon=False)
    plt.show()
# plot_8()
def generate_plot(K,X,T,n):
    Y = np.zeros([n,len(K)]) #NOKO
    X = X.detach().numpy()  #hnn
    T = T.detach().numpy() # true
    Y[0,:]=T[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    plot_8(T,Y,X)
    # print()


class kepler(nn.Module):
    dim = 4
    def forward(self, t, X):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(X)
        x,y,a,b = X[:,0],X[:,1],X[:,2],X[:,3]
        dx[:, 0] = a
        dx[:, 1] = b
        dist = (x**2+y**2)**1.5
        dx[:, 2] = -x/dist
        dx[:, 3] = -y/dist
        return dx

def generate():
    n = 300
    y0 = torch.tensor([[1.0, 0.0, 0., 0.9]])
    t = torch.linspace(0, 12, n)
    y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
    # np.save('./data/true_y_9_150',y)
    np.save('./data/true_y_12_300', y)
generate()