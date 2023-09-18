import math

import numpy as np
import torch

from functions import *


class NC_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(NC_Net, self).__init__()
        # torch.manual_seed(2)
        self.recurrent_kernel = nn.Linear(n_input, n_output, bias=False)

    def forward(self, data):
        return self.recurrent_kernel(data)

class lift_Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
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
            nn.Linear(n_hidden,n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)


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
            nn.Linear(n_hidden,n_output)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)



true_y = np.load('./data/3d_24body.npy')
true_y = torch.from_numpy(true_y)

'''
For learning 
'''
sigma = 0.03
lift_d = 80-72 # 70-60
dim = 24*3
D_in = dim+lift_d # input dimension
H1 = 64 # 64

out_iters = 0
eigen_list=[]
Loss_list = []

while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = NC_Net(D_in, D_in)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    Enc = lift_Net(dim,H1,dim+lift_d)
    Dec = Decoder(dim+lift_d,H1,dim)

    i = 0
    max_iters = 10000
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in Enc.parameters()]+
                                 [i for i in Dec.parameters()], lr=learning_rate)
    while i < max_iters:
        # break
        lift_y = Enc(y)
        dec_y = Dec(lift_y)
        X1, X2, X3, X4 = lift_y[:1800 - 3], lift_y[1:1800 - 2], lift_y[2:1800 - 1], lift_y[3:1800]
        y1, y2, y3, y4 = y[:1800-3], y[1:1800-2], y[2:1800-1], y[3:1800]
        loss = torch.sum((X2-model(X1))**2)+torch.sum((X3-model(model(X1)))**2)+torch.sum((X4-model(model(model(X1))))**2)\
               +torch.sum((dec_y-y)**2) \
                +torch.sum((Dec(model(X1))-y2)**2)+torch.sum((Dec(model(model(X1)))-y3)**2)+torch.sum((Dec(model(model(model(X1))))-y4)**2)
        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        Loss_list.append(loss.item())
        if loss.item() == min(Loss_list):
            K = model.recurrent_kernel.weight.detach().numpy()
        i += 1
    stop = timeit.default_timer()
    # K = model.recurrent_kernel.weight.detach().numpy()
    Loss = torch.tensor(Loss_list)
    print('Best results: ({},{})'.format(torch.argmin(Loss),torch.min(Loss)))
    n = len(true_y)

    lift_y = Enc(true_y)
    dec_y = Dec(lift_y)[:-1]
    y = y.detach().numpy()
    dec_y=dec_y.detach().numpy()

    Y = np.zeros([n,len(K)]) #NOKO
    Y[0,:]=lift_y.detach().numpy()[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()
    np.save('./data/dlko_{}_{}_{}'.format(sigma,H1,max_iters),pred_y)
    Y = pred_y
    plt.subplot(121)
    plt.scatter(true_y[:,0],true_y[:,1])
    plt.title('Truth')
    plt.subplot(122)
    plt.scatter(pred_y[:,0],pred_y[:,1])
    plt.title('DLKO')
    print('\n')
    print("Total time: ", stop - start)
    out_iters += 1


plt.show()