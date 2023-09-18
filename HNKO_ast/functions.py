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


class K_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(K_Net, self).__init__()
        # torch.manual_seed(2)
        self.recurrent_kernel = nn.Linear(n_input, n_output, bias=False)
        geotorch.orthogonal(self.recurrent_kernel, "weight")
        self.reset_parameters()

    def reset_parameters(self):
        # The manifold class is under `layer.parametrizations.tensor_name[0]`
        M = self.recurrent_kernel.parametrizations.weight[0]
        # Every manifold has a convenience sample method, but you can use your own initializer
        self.recurrent_kernel.weight = M.sample("uniform")

    def forward(self, data):
        return self.recurrent_kernel(data)

class K2_Net(torch.nn.Module):
    def __init__(self, dim1,dim2):
        super(K2_Net, self).__init__()
        # torch.manual_seed(2)
        self.layer1= nn.Linear(dim1,dim1, bias=False)
        self.layer2 = nn.Linear(dim2, dim2, bias=False)
        geotorch.orthogonal(self.layer1, "weight")
        geotorch.orthogonal(self.layer2, "weight")
        self.reset_parameters()

    def reset_parameters(self):
        # The manifold class is under `layer.parametrizations.tensor_name[0]`
        M1 = self.layer1.parametrizations.weight[0]
        M2 = self.layer2.parametrizations.weight[0]
        # Every manifold has a convenience sample method, but you can use your own initializer
        self.layer1.weight = M1.sample("uniform")
        self.layer2.weight = M2.sample("uniform")

    def forward(self, data):
        W1 = self.layer1.weight
        W2 = self.layer2.weight
        T = torch.kron(W1,W2)
        return torch.mm(data,T.T)
        # return self.recurrent_kernel(data)

class kdv(nn.Module):
    def forward(self, t, x):
        # x: [b_size, d_dim]
        v = torch.tensor(1.0)
        c = torch.tensor(2.0)
        return -1.0/torch.cosh(-math.sqrt(0.5)*(x-2*t))**2

# func = kdv()
# x = torch.linspace(-50,50,100)
# t = torch.linspace(0,5,100)
# y = func(10.0,x)
# plt.plot(x,y)
# plt.show()

def state_err(true,pred):
    # data size (501,64)
    err = np.sqrt(np.mean((true-pred)**2,axis=1))
    return err

def mass_err(true,pred):
    # data size (501,64)
    true_mass = np.sum(true,axis=1)
    pred_mass = np.sum(pred,axis=1)
    err = np.abs(true_mass-pred_mass)
    return pred_mass


def energy_err(true,pred):
    # data size (501,64)
    true_e = np.sum(true**2,axis=1)
    pred_e = np.sum(pred**2,axis=1)
    err = np.abs(true_e-pred_e)
    return pred_e

def error_case(true,pred,case):
    if case==1:
        return state_err(true,pred)
    if case==2:
        return mass_err(true,pred)
    if case==3:
        return energy_err(true,pred)

def pred_data(K,true_y):
    pred_y = np.zeros_like(true_y)
    pred_y[0,:]=true_y[0,:]
    for i in range(len(true_y)-1):
        y = pred_y[i,:].reshape(-1,1)
        pred_y[i+1,:]=np.matmul(K,y).T[0]
    return pred_y

def plot(true,dmd,koop,rangeX,rangeT):
    plt.figure(figsize=(8, 8))
    plt.subplot(151)
    plt.imshow(true, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('True')

    plt.subplot(152)
    plt.imshow(koop, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('HNKO')

    plt.subplot(153)
    plt.imshow(dmd, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('DMD')

    plt.subplot(154)
    error = np.sum((dmd-true)**2,axis=1)
    plt.plot(np.arange(len(error)),error)
    # plt.imshow(dmd, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.axis('auto')
    print('DMD MSE:',error[400:].mean())
    plt.title('DMD')

    plt.subplot(155)
    error = np.sum((koop-true)**2,axis=1)
    plt.plot(np.arange(len(error)),error)
    # plt.imshow(koop, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.axis('auto')
    plt.title('HKNO')
    print('HNKO MSE:',error[400:].mean())
    plt.show()

def plot1(true,koop,rangeX,rangeT):
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.imshow(true, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('True')

    plt.subplot(122)
    error = np.sum((koop-true)**2,axis=1)
    plt.plot(np.arange(len(error)),error)
    # plt.imshow(koop, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.axis('auto')
    # plt.title('HKNO')
    plt.show()

def plot8():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    if 0==0:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
        # matplotlib.rcParams['text.usetex'] = True
    fontsize = 12
    labelsize = 12
    data = np.load('./data/samples_64_501.npy', allow_pickle=True).item()
    t = data['t']
    x = data['x']
    true_y = data['u_x']
    item_list = ['Noisy Data','NODE','DMD','HNKO']
    data_list = ['noisy',  'node', 'dmd','hkno']
    rangeX, rangeT = 50, 600
    plt.figure(dpi=150)
    # plt.figure(figsize=(20,8),dpi=150)
    for i in range(4):
        plt.subplot(2,4,i+1)
        sol = np.load('./data/{}_64_501_0.03.npy'.format(data_list[i]))
        plt.imshow(sol.T, extent=[0, rangeT, 0, rangeX], cmap='bwr', vmin=-0.4, vmax=0.4)  # test sol[::-1, :]
        if i==3:
            pass
            # c = plt.colorbar()
            # c.set_ticks([-0.4,0,0.4])
        if i==0:
            plt.ylabel(r'$x$',fontsize=fontsize)
        plt.xlabel('Time', fontsize=fontsize, labelpad=-10)
        plt.xticks([0,600])
        plt.yticks([0,50])
        plt.axis('auto')
        plt.title('{}'.format(item_list[i]),fontsize=fontsize)

        if i == 0:
            plt.subplot(2,4,5)
            for j in range(1,4):
                sol = np.load('./data/{}_64_501_0.03.npy'.format(data_list[j]))
                plt.plot(x,sol[400],label='{}'.format(item_list[j]),color=colors[j-1])
            plt.plot(x, true_y[400], label='Truth', color='black',ls=(0,(3,3)))
            plt.xticks([0,50])
            plt.xlabel(r'$x$',fontsize=fontsize)
            plt.yticks([0,0.4])
            plt.ylabel(r'$u(400,x)$',fontsize=fontsize)

        if i!=0:
            plt.subplot(2,4,i+1+4)
            for j in range(1,4):
                sol = np.load('./data/{}_64_501_0.03.npy'.format(data_list[j]))
                err = error_case(true_y,sol,i)
                plt.plot(t,err,color=colors[j-1],label='{}'.format(item_list[j]))
            err = error_case(true_y, true_y, i)
            plt.plot(t, err, color='black', label='Truth', ls=(0, (3, 3)))
            # plt.legend()
            plt.xticks([0,600])

            plt.axis('auto')
            if i == 1:
                plt.ylabel('State error', fontsize=labelsize,labelpad=-20)
                # plt.title('State',fontsize=fontsize)
                plt.xlabel('Time',fontsize=fontsize)
                plt.yticks([0,0.15],['0','0.15'])
            if i == 2:
                plt.ylabel('Mass',fontsize=labelsize,labelpad=-10)
                plt.xlabel('Time',fontsize=fontsize)
                plt.yticks([0,2.0],['0','2'])
            if i == 3:
                plt.ylabel('Energy',fontsize=labelsize,labelpad=-15)
                plt.xlabel('Time',fontsize=fontsize)
                plt.yticks([0,0.5],['0','0.5'])
                plt.legend(ncol=1,bbox_to_anchor=[0,-0.3],fontsize=8,frameon=False)


    plt.show()

def subsubplot():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    if 0 == 0:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
        # matplotlib.rcParams['text.usetex'] = True
    fontsize = 12
    labelsize = 12
    data = np.load('./data/samples_64_501.npy', allow_pickle=True).item()
    t = data['t'][:250]
    print(t.shape)
    x = data['x']
    true_y = data['u_x'][:250]
    item_list = ['Noisy Data', 'NODE', 'DMD', 'HNKO']
    data_list = ['noisy', 'node', 'dmd', 'hkno']
    for j in range(1, 4):
        if j !=2:
            sol = np.load('./data/{}_64_501_0.03.npy'.format(data_list[j]))[:250]
            err = error_case(true_y, sol, 2)
            plt.plot(t, err, color=colors[j - 1], label='{}'.format(item_list[j]))
    err = error_case(true_y, true_y, 2)
    plt.plot(t, err, color='black', label='Truth', ls=(0, (3, 3)))
    # plt.legend()
    plt.xticks([0, 300])
    plt.yticks([1.9, 2.4])

    plt.axis('auto')
    plt.show()

# subsubplot()
# plot8()