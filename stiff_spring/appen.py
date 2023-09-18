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



# H=k*q^2/2+p^2/(2m)
class spring(nn.Module):
    def __init__(self, k,m):
        super(spring, self).__init__()
        self.k = torch.tensor(k, requires_grad=True)
        self.m = torch.tensor(m, requires_grad=True)

    def forward(self, t, X):
        # k,m = 1000.0,3000  # stiff case
        # k,m = 1.0,1.0 # normal case
        dx = torch.zeros_like(X)
        q,p = X[:,0],X[:,1]
        dx[:, 0] = p/self.m
        dx[:, 1] = -q*self.k
        return dx

class stiff(nn.Module):
    def forward(self,t,X):
        dx = torch.zeros_like(X)
        x = X[:,0]
        dx[:,0] = -1000*x+3000-2000*math.exp(-t)
        return dx

def ana_orbit(k,m):
    a,b=np.sqrt(2/k),np.sqrt(2*m)
    s = np.linspace(0,2*np.pi,1000)
    r = a*b/np.sqrt((b*np.cos(s))**2+(a*np.sin(s))**2)
    plt.scatter(r*np.cos(s),r*np.sin(s))
    # plt.show()


# ana_orbit(1000,3000)

def lift(y):
    Z = torch.zeros([len(y),3])
    # r = torch.max(y)**2+torch.min(torch.abs(y))**2
    r=torch.max(torch.sum(y**2,dim=1))
    for i in range(len(Z)):
        Z[i,:2]=y[i,:]
        z = torch.sqrt(r-torch.sum(y[i,:]**2))
        if y[i,0]>=0:
            Z[i,2]=z
        else:
            Z[i,2]=-z
    return Z

def generate():
    y0 = torch.tensor([[1.0, 0.0]])
    # t = torch.linspace(0, 3, 100)  # train
    t = torch.linspace(0, 9, 300)  # pred
    stiff_list = torch.linspace(1.0, 200, 40)
    data = np.zeros([40, 300, 4])  # state,vector field
    for i in range(40):
        a = stiff_list[i]
        k = torch.pow(a, 1)
        m = torch.pow(a, 1)
        func = spring(k, m)
        y = odeint(func, y0, t, atol=1e-8, rtol=1e-8)[:, 0, :]
        dy = func(0.0,y)
        data[i,:, :2] = y.detach().numpy()
        data[i,:, 2:] = dy.detach().numpy()
        print(i)
    # np.save('./data/train_y_3_100',data)
    np.save('./data/true_y_9_300', data)
# generate()

def state_err(true,pred):
    # size (:,2)
    err = np.sqrt(np.sum((true-pred)**2,axis=1))
    return err

def energy(data,k,m):
    # size (:,2)
    e = 0.5*k*data[:,0]**2+0.5*data[:,1]**2/m
    return e

def unify(x):
    x = x / np.max(x)
    mse = np.mean(np.abs(x[:, 0]))
    return mse

true = np.load('./data/true_y_9_300.npy')
print(true.shape)

def appen_plot():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction']='in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 15
    labelsize=12
    low = np.load('./data/true_y_9_300.npy')
    true = np.load('./data/true_y_9_300.npy')[:,:,:2]
    # lift_y = lift(torch.from_numpy(true)).detach().numpy()

    cor = []
    for i in range(40):
        lift_y = lift(torch.from_numpy(true[i,:])).detach().numpy()
        lift_y = lift_y/np.max(lift_y)
        err = np.mean(np.sqrt(lift_y[:,0]**2+lift_y[:,2]**2))
        cor.append(err)
    cor = np.array(cor)
    stiff_list = np.linspace(1.0, 200, 40)
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.88, hspace=0.27, wspace=0.3)
    plt.subplot(131)
    co1 = 'tab:blue'
    plt.plot(stiff_list,np.power(cor,1.0),c=co1)
    plt.ylim(0,1)
    plt.xlabel('SC',fontsize=fontsize)
    plt.ylabel('MSE',fontsize=fontsize)
    plt.yticks([0, 0.5, 1.0],  fontsize=fontsize)
    plt.xticks([0, 100, 200], fontsize=fontsize)
    # plt.scatter(stiff_list[-1], np.power(cor, 1)[-1], s=25, c='r')

    plt.subplot(132)
    # phase plot
    k = -1
    plt.plot(true[k, :207, 0], true[k, :207, 1], color='tab:red', ls=(0, (5, 5)), label='Truth')
    plt.xlabel(r'$x$',fontsize=fontsize)
    # plt.ylabel(r'$y$',fontsize=fontsize)
    plt.yticks([-200,0,200],fontsize=labelsize)
    plt.xticks([-20,0,20],fontsize=labelsize)
    plt.xlabel(r'$q$',fontsize=fontsize,labelpad=-1)
    plt.ylabel(r'$p$',fontsize=fontsize,labelpad=-15)
    plt.title('SC=200',fontsize=fontsize)
    # plt.legend()

    ax = fig.add_subplot(133,projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    # ax.view_init(24,40) #rotate
    # ax.set_xticks([-4,-2,0, 2,4],['','','','',''])
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_zlim(-100,100)
    ax.set_zticks([])
    ax.set_xlabel(r'$q$',fontsize=fontsize)
    ax.set_ylabel(r'$p$',fontsize=fontsize)
    ax.set_zlabel(r'$z$',fontsize=fontsize)
    plt.title('SC=200', fontsize=fontsize)
    lift_y = lift(torch.from_numpy(true[k, :])).detach().numpy()
    ax.plot(lift_y[:,0],lift_y[:,1],lift_y[:,2], color='tab:red', ls=(0, (5, 5)), label='Truth')
    # plt.xticks([])
    # plt.yticks([])
    # ax.set_zticks([])
    # null = [0]*len(lift_y[:,2])
    # ax.plot(lift_y[:,0],lift_y[:,1],null)
    # ax.plot(low[:,0],low[:,1],null)
    # plt.xlim(-10,10)
    plt.show()

appen_plot()
