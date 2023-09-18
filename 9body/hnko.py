import math

import matplotlib.pyplot as plt
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
    def __init__(self, n_input, n_hidden, n_output,k,q):
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
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
            nn.Linear(n_hidden,n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.k = torch.tensor(107.0, requires_grad=True)
        # self.k = torch.tensor(k, requires_grad=True)
        self.v = torch.randn([n_output,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out

class Decoder(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.Tanh(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.Tanh(),
            nn.Linear(n_hidden,n_output)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)





true_y = np.load('./data/true_9_5_100.npy')[:40]
true_y = torch.from_numpy(true_y)
# plt.plot(np.arange(len(true_y)),true_y[:,0])
# plt.scatter(true_y[:,0],true_y[:,1])
# print(true_y.shape)



'''
For learning 
'''
sigma = 0.03
lift_d = 30 # 19
q = 33  # 13
dim = 9*4
D_in = dim+lift_d # input dimension
H1 = 128
D_out = dim+lift_d # output dimension
k = torch.max(torch.sum(true_y**2,dim=1))

out_iters = 0
eigen_list=[]
while out_iters < 1:
    break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    g1 = lift_Net(dim,H1,dim+lift_d,k,q)
    Dec = Decoder(dim+lift_d,H1,dim)

    i = 0
    max_iters = 10000
    learning_rate = 0.0005
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
        loss = torch.sum((X2-model(X1))**2) \
               +torch.sum((dec_y-y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)
        print(out_iters,i, "loss=", loss.item(),g1.k)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # if loss<=1:
        #     break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()
    # X1 = X1.T.detach().numpy()
    # X2 = X2.T.detach().numpy()
    # K = np.matmul(X2,np.linalg.pinv(X1))

    # y0=torch.from_numpy(fix_config()).view([1,12])
    n = len(true_y)
    # n = 100
    # t=torch.linspace(0,5*4,n)
    # true_y=odeint(threebody(),y0,t,atol=1e-8,rtol=1e-8)[:,0,:]

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
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()
    # np.save('./data/hnko_66_33_5_100_{}'.format(sigma),pred_y)
    Y = pred_y
    plt.subplot(121)
    plt.scatter(true_y[:,0],true_y[:,1])
    # plt.scatter(true_y[:,2],true_y[:,3])
    # plt.scatter(true_y[:,4],true_y[:,5])
    # plt.scatter(true_y[:,6],true_y[:,7])
    # plt.scatter(true_y[:,8],true_y[:,9])
    # plt.scatter(true_y[:,10],true_y[:,11])
    plt.title('Truth')
    # plt.plot(dec_y[:,0],dec_y[:,1])
    plt.subplot(122)
    plt.scatter(pred_y[:,0],pred_y[:,1])
    # plt.scatter(pred_y[:,2],pred_y[:,3])
    # plt.scatter(pred_y[:,4],pred_y[:,5])
    # plt.scatter(pred_y[:,6],pred_y[:,7])
    # plt.scatter(pred_y[:,8],pred_y[:,9])
    # plt.scatter(pred_y[:,10],pred_y[:,11])
    plt.title('HNKO')
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
    out_iters += 1


plt.show()