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

def appen_plot():
    low = np.load('./data/true_y_9_300.npy')[0,:205,:2]
    true = np.load('./data/true_y_9_300.npy')[-1,:205,:2]
    lift_y = lift(torch.from_numpy(true)).detach().numpy()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(lift_y[:,0],lift_y[:,1],lift_y[:,2])

    null = [0]*len(lift_y[:,2])
    ax.plot(lift_y[:,0],lift_y[:,1],null)
    ax.plot(low[:,0],low[:,1],null)
    # plt.xlim(-10,10)
    plt.show()

def compare():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction']='in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 15
    labelsize= 15
    hnn = np.load('./data/hnn_9_300.npy')[:,:,:]
    hkno = np.load('./data/hkno_9_300.npy')[:,:,:]
    steer = np.load('./data/steer_9_300.npy')[:,:,:]
    true = np.load('./data/true_y_9_300.npy')[:,:,:2]
    train = np.load('./data/train_y_3_100.npy')
    stiff_list = np.linspace(1.0, 200, 40)

    cor = []
    hnn_err = []
    steer_err = []
    hkno_err = []
    for i in range(40):
        err = unify(true[i,:])
        cor.append(err)
        hnn_err.append(np.mean(state_err(true[i,:],hnn[i,:])))
        steer_err.append(np.mean(state_err(true[i,:],steer[i,:])))
        hkno_err.append(np.mean(state_err(true[i],hkno[i])))
    cor = np.array(cor)

    co1,co2 = 'tab:blue','tab:red'
    ax1 = plt.subplot(151)
    plt.plot(stiff_list,np.power(cor,1.0),c=co1)
    # plt.scatter(stiff_list[0],np.power(cor,0.3)[0],s=25,c='r')
    # plt.scatter(stiff_list[-1], np.power(cor, 0.3)[-1], s=25, c='r')
    plt.xlabel('SI',fontsize=fontsize)
    plt.ylabel('MSE',c=co1,fontsize=fontsize)
    plt.xticks([0,100,200],fontsize=labelsize)
    plt.yticks([0,0.3,0.6],color=co1,fontsize=labelsize)
    ax2 = ax1.twinx()
    plt.plot(stiff_list,np.array(hnn_err),c=co2,ls='dashed',label='HNN')
    plt.plot(stiff_list, np.array(steer_err), c=co2,ls='dotted',label='STEER')
    plt.plot(stiff_list, np.array(hkno_err), c=co2,ls='solid',label='HNKO')
    plt.ylabel('LTSE',c=co2,fontsize=fontsize)
    plt.yticks([0,175],color=co2,fontsize=labelsize)
    plt.legend(loc=9)


    plt.subplot(152)
    # phase plot
    k = 0
    plt.plot(hnn[k,:,0],hnn[k,:,1],color=colors[0],label='HNN')
    plt.plot(steer[k,:,0],steer[k,:,1],color=colors[1],label='STEER')
    plt.plot(hkno[k, :, 0], hkno[k, :, 1], color=colors[2], label='HNKO')
    plt.plot(true[k, :207, 0], true[k, :207, 1], color='black', ls=(0, (5, 5)), label='Truth')
    plt.xlabel(r'$x$',fontsize=fontsize)
    # plt.ylabel(r'$y$',fontsize=fontsize)
    plt.yticks([-1,0,1],fontsize=labelsize)
    plt.xticks([-1,0,1],fontsize=labelsize)
    plt.title('SI=1',fontsize=fontsize)
    # plt.legend()

    plt.subplot(153)
    # phase plot
    k = -1
    plt.plot(hnn[k,:,0],hnn[k,:,1],color=colors[0],label='HNN')
    plt.plot(steer[k,:,0],steer[k,:,1],color=colors[1],label='STEER')
    plt.plot(hkno[k, :, 0], hkno[k, :, 1], color=colors[2], label='HNKO')
    plt.plot(true[k, :207, 0], true[k, :207, 1], color='black', ls=(0, (5, 5)), label='Truth')
    plt.xlabel(r'$x$',fontsize=fontsize)
    # plt.ylabel(r'$y$',fontsize=fontsize)
    plt.yticks([-300,0,200],fontsize=labelsize)
    plt.xticks([-1,0,1],fontsize=labelsize)
    plt.title('SI=1', fontsize=fontsize)
    # plt.legend()

    plt.subplot(154)
    k = -1
    # plt.plot(train[k, :, 0], train[k, :, 1])
    t = np.linspace(0,9,300)
    plt.plot(t, state_err(true[k],hnn[k]),c=colors[0],label='HNN')
    plt.plot(t, state_err(true[k], steer[k]),c=colors[1],label='STEER')
    plt.plot(t, state_err(true[k], hkno[k]),c=colors[2],label='HNKO')
    plt.plot(t, state_err(true[k], true[k]), c='black', ls=(0,(5,5)),label='Truth')
    plt.legend(ncol=4,bbox_to_anchor=[0,-0.3],fontsize=15,frameon=False)
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel('State error',fontsize=fontsize)
    plt.xticks([0, 9],fontsize=labelsize)
    plt.yticks([0,400],fontsize=labelsize)

    plt.subplot(155)
    plt.plot(t,energy(hnn[-1], 200, 200), c=colors[0], label='HNN')
    plt.plot(t,energy(steer[-1], 200, 200), c=colors[1], label='STEER')
    plt.plot(t,energy(hkno[-1], 200, 200), c=colors[2], label='HNKO')
    plt.plot(t, energy(true[-1], 200, 200), c='black', ls=(0,(5,5)), label='Truth')
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel('Energy',fontsize=fontsize)
    plt.xticks([0,9],fontsize=labelsize)
    plt.yticks([0,100,500],fontsize=labelsize)
    # plt.legend()

    plt.show()

# compare()