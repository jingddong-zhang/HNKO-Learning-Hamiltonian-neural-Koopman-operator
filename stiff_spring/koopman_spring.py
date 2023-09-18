import matplotlib.pyplot as plt

from functions import *


import math

import numpy as np
import torch

from functions import *

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

class lift_Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output,k):
        super(lift_Net, self).__init__()
        # torch.manual_seed(2)
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.k = torch.tensor(k, requires_grad=True)
        self.v = torch.randn([2+lift_d,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out

class Decoder(nn.Module):

    def __init__(self,n_hidden):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2+lift_d, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)


n = 100
# y0 = torch.tensor([[np.sqrt(2/1000),0.0]])
# t = torch.linspace(0, 10, n)
# # y = odeint(pendulum(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
# true_y = odeint(spring(), y0, t, atol=1e-3, rtol=1e-3)[:,0,:]


train_data = np.load('./data/train_y_3_100.npy')
true_y = torch.from_numpy(train_data[10,:,:2])
k = torch.max(torch.sum(true_y**2,dim=1))
true_y *= 1/k
# plt.plot(true_y[:,0],true_y[:,1])
'''
For learning 
'''
sigma = 0.0
lift_d = 1
q = 2
D_in = 2+lift_d # input dimension
H1 = 12
D_out = 2+lift_d # output dimension


out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    g1 = lift_Net(2,H1,2+lift_d,torch.max(torch.sum(true_y**2,dim=1)))
    Dec = Decoder(H1)

    i = 0
    max_iters = 5000
    learning_rate = 0.005
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in g1.parameters()]+[g1.k]+[g1.v]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)
    while i < max_iters:
        # break
        lift_y = g1(y)
        dec_y = Dec(lift_y)
        X1,X2 = lift_y[:-1],lift_y[1:]
        v = g1.v/torch.sqrt(torch.sum(g1.v**2,dim=0))
        V = torch.mm(v.T,v)-torch.eye(q)
        # embed_y = torch.cat((y,g1(y)),dim=1)
        # loss = torch.sum((X2-model(X1))**2) \
        loss = torch.sum((X2-model(X1))**2) \
               +torch.sum((dec_y-y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)

        print(out_iters,i, "loss=", loss.item(),g1.k)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()
    # X1 = X1.T.detach().numpy()
    # X2 = X2.T.detach().numpy()
    # K = np.matmul(X2,np.linalg.pinv(X1))

    y0=torch.tensor([[1.0,0.0]])
    t=torch.linspace(0,10,n)
    # true_y=odeint(spring(),y0,t,atol=1e-8,rtol=1e-8)[:,0,:]

    lift_y = g1(true_y)
    # XX1,XX2 = re_data(true_y,lift_y)
    dec_y = Dec(lift_y)[:-1]
    y = y.detach().numpy()
    dec_y=dec_y.detach().numpy()

    Y = np.zeros([n,len(K)]) #NOKO
    Y[0,:]=lift_y.detach().numpy()[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()*k.item()
    # np.save('./data/edmd_0.03',pred_y)
    Y = pred_y
    # plt.subplot(131)
    # plt.plot(true_y[:,0],true_y[:,1])
    plt.plot(dec_y[:,0],dec_y[:,1])
    # plt.plot(pred_y[:,0],pred_y[:,1])
    # plt.subplot(132)
    # plt.plot(true_y[:,2],true_y[:,3])
    # plt.plot(dec_y[:,2],dec_y[:,3])
    # plt.plot(pred_y[:,2],pred_y[:,3])
    # plt.subplot(133)
    # plt.plot(true_y[:,2],true_y[:,3])
    # plt.plot(dec_y[:,2],dec_y[:,3])
    # plt.plot(pred_y[:,2],pred_y[:,3])
    # plot_4(true_y.detach().numpy(),pred_y,true_y.detach().numpy())
    # pred_pendulum(K,X1,XX1,50)
    # generate_plot(K,X1,XX1,49)
    # print('test err',torch.sum((X2 - model(X1))**2))
    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1


plt.show()