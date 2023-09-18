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

        self.k = torch.tensor(180.0, requires_grad=True)
        # self.k = torch.tensor(k/10, requires_grad=True)
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
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
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


true_y = np.load('./data/3d_24body_full.npy')
true_y = torch.from_numpy(true_y)
# plt.plot(np.arange(len(true_y)),true_y[:,0])
# N = 6
# plt.scatter(true_y[:,N*4],true_y[:,N*4+1])
# print(true_y.shape)



'''
For learning 
'''
sigma = 0.03
lift_d = 154-144 # 70-60
q = 80  # 40
dim = 24*6
D_in = dim+lift_d # input dimension
H1 = 64 # 64
D_out = dim+lift_d # output dimension
k = torch.max(torch.sum(true_y**2,dim=1))
# print(k)
out_iters = 0
eigen_list=[]
Loss_list = []

while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    sigma = 0.01*(out_iters+1)
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
        # X1,X2 = lift_y[:-1],lift_y[1:]
        X1, X2, X3, X4 = lift_y[:1800-3], lift_y[1:1800-2], lift_y[2:1800-1], lift_y[3:1800]
        v = g1.v/torch.sqrt(torch.sum(g1.v**2,dim=0))
        V = torch.mm(v.T,v)-torch.eye(q)
        # embed_y = torch.cat((y,g1(y)),dim=1)
        loss = torch.sum((X2-model(X1))**2)+torch.sum((X3-model(model(X1)))**2)+torch.sum((X4-model(model(model(X1))))**2) \
               +torch.sum((dec_y-y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)
        print(out_iters,i,'sigma={}'.format(sigma), "loss={}".format(loss.item()),g1.k)
        Loss_list.append(loss.item())
        if loss.item() == min(Loss_list):
            K = model.recurrent_kernel.weight.detach().numpy()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # if loss<=5:
        #     optimizer = torch.optim.Adam(
        #         [i for i in model.parameters()] + [i for i in g1.parameters()] + [g1.k] + [g1.v] + \
        #         [i for i in Dec.parameters()], lr=1e-4)
        #     break
        i += 1
    stop = timeit.default_timer()
    Loss = torch.tensor(Loss_list)
    print('Best results: ({},{})'.format(torch.argmin(Loss),torch.min(Loss)))
    # K = model.recurrent_kernel.weight.detach().numpy()

    # K = torch.kron(model.layer1.weight, model.layer2.weight).detach().numpy()
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
    np.save('./data/hnko_154_{}_{}_{}_{}'.format(q,sigma,H1,max_iters),pred_y)
    # pred_y = torch.from_numpy(np.load('./data/hnko_154_80_0.02_64.npy'))
    Y = pred_y
    fig = plt.figure()
    plt.subplot(121)
    plt.scatter(true_y[:,0],true_y[:,1])
    plt.title('Truth')
    plt.subplot(122)
    plt.scatter(pred_y[:,0],pred_y[:,1])
    plt.title('HNKO')
    # print('test err',torch.sum((X2 - model(X1))**2))
    print('\n')
    print("Total time: ", stop - start)
    out_iters += 1


plt.show()