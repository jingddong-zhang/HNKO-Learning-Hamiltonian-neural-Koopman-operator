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
fontsize = 20
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


def rotate(n,a,sigma):
    np.random.seed(13)
    A = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    x0 = np.array([1.0,0.0])
    X = np.zeros([n,2])
    X[0,:]=x0
    for i in range(n-1):
        x = X[i,:].reshape(2,1)
        X[i+1,:]=np.matmul(A,x).T[0]
    XX = X+np.random.normal(0,sigma,X.shape)
    X1,X2 = XX[:-1].T,XX[1:].T
    return X,XX

def pred_rotate(K,n,a,sigma):
    A = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    np.random.seed(13)
    x0 = np.array([1.0,0.0])
    def generate(A):
        X = np.zeros([n,2])
        X[0,:]=x0
        for i in range(n-1):
            x = X[i,:].reshape(2,1)
            X[i+1,:]=np.matmul(A,x).T[0]
        return X
    X = generate(A)
    XX = X+np.random.normal(0,sigma,X.shape)
    X1,X2 = XX[:-1].T,XX[1:].T
    B = np.matmul(X2, np.linalg.pinv(X1))
    Y = generate(B)
    YY = generate(K)
    plt.scatter(X[:,0],X[:,1],color=colors[0],label='True')
    # plt.scatter(XX[:, 0], XX[:, 1],color=colors[0],label='noisy data')
    plt.scatter(Y[:, 0], Y[:, 1],color=colors[1],label='DMD')
    plt.scatter(YY[:, 0], YY[:, 1],color=colors[2], label='NOKO')
    # plt.title(r'$\sigma^2$=0.1',fontsize=fontsize)
    plt.legend()
    plt.show()
    return X,XX

def plot(n,a):
    A = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    np.random.seed(13)
    x0 = np.array([1.0,0.0])
    def generate(A):
        X = np.zeros([n,2])
        X[0,:]=x0
        for i in range(n-1):
            x = X[i,:].reshape(2,1)
            X[i+1,:]=np.matmul(A,x).T[0]
        return X
    X = generate(A)
    plt.subplot(121)
    X1,X2 = X[:-1].T,X[1:].T
    B = np.matmul(X2, np.linalg.pinv(X1))
    Y = generate(B)
    YY = generate(NOKO(0.0))
    plt.plot(X[:,0],X[:,1],color=colors[0],label='True',alpha=0.5,lw=3)
    plt.plot(Y[:, 0], Y[:, 1],color=colors[2],label='DMD',ls='dashed',lw=2)
    plt.scatter(YY[:, 0], YY[:, 1],color=colors[1], label='HNKO',lw=0.1)
    plt.title(r'$\sigma^2$=0.0',fontsize=20)
    plt.xticks([-1,0,1],fontsize=fontsize)
    plt.yticks([-1,0,1],fontsize=fontsize)
    plt.xlabel(r'$x$',fontsize=fontsize)
    plt.ylabel(r'$y$',fontsize=fontsize)
    plt.legend(frameon=False)
    plt.subplot(122)
    XX = X+np.random.normal(0,sigma,X.shape)
    X1,X2 = XX[:-1].T,XX[1:].T
    B = np.matmul(X2, np.linalg.pinv(X1))
    Y = generate(B)
    YY = generate(NOKO(0.1))
    plt.plot(X[:,0],X[:,1],color=colors[0],label='True',alpha=0.5,lw=3)
    plt.plot(Y[:, 0], Y[:, 1],color=colors[2],label='DMD',ls='dashed',lw=2)
    plt.scatter(YY[:, 0], YY[:, 1],color=colors[1], label='HNKO',lw=0.1)
    plt.title(r'$\sigma^2$=0.1',fontsize=20)
    plt.xlabel(r'$x$',fontsize=fontsize)
    plt.ylabel(r'$y$',fontsize=fontsize)
    plt.xticks([-1,0,1],fontsize=fontsize)
    plt.yticks([-1,0,1],fontsize=fontsize)
    plt.legend(frameon=False,loc=4)
    plt.show()
    return X,XX


def generate(n,A):
    x0=np.array([1.0,0.0])
    X = np.zeros([n,2])
    X[0,:]=x0
    for i in range(n-1):
        x = X[i,:].reshape(2,1)
        X[i+1,:]=np.matmul(A,x).T[0]
    return X

sigma = 0.1
X,XX=rotate(100,2*np.pi/100,sigma)
# X1,X2=XX[:-1].T,XX[1:].T
# B=np.matmul(X2,np.linalg.pinv(X1))
# Y = generate(50,B)
# plt.scatter(X[:,0],X[:,1])
# plt.scatter(Y[:,0],Y[:,1])
# plt.show()
X = torch.from_numpy(X)
XX = torch.from_numpy(XX)
X1,X2=X[:-1],X[1:]
XX1,XX2,XX3=XX[:-2],XX[1:-1],XX[2:]



'''
For learning 
'''

D_in = 2 # input dimension
D_out = 2 # output dimension
def NOKO(sigma):
    X,XX=rotate(100,2*np.pi/100,sigma)
    X=torch.from_numpy(X)
    XX=torch.from_numpy(XX)
    X1,X2=X[:-1],X[1:]
    XX1,XX2,XX3=XX[:-2],XX[1:-1],XX[2:]
    out_iters = 0
    eigen_list=[]
    while out_iters < 1:
        # break
        start = timeit.default_timer()
        torch.manual_seed(out_iters*9)
        # np.random.seed(out_iters)
        model = K_Net(D_in, D_out)
        i = 0
        max_iters = 1000
        learning_rate = 0.05
        optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
        while i < max_iters:
            # break
            loss = torch.sum((XX2-model(XX1))**2)+torch.sum((XX3-model(model(XX1)))**2)
            print(out_iters,i, "loss=", loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if loss<=1e-6:
                break
            i += 1
        stop = timeit.default_timer()
        K = model.recurrent_kernel.weight.detach().numpy()
        # pred_rotate(K,100,2*np.pi/100,sigma)
        print('test err',torch.sum((X2 - model(X1))**2))
        print('\n')
        print("Total time: ", stop - start)
        # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
        out_iters += 1
        return K

plot(100,2*np.pi/100)