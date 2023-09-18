import numpy as np
import torch

from functions import *

import argparse
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--neurons', type=int, default=32)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--layers', type=int, default=2)
args = parser.parse_args()


class HNN(nn.Module):

    def __init__(self,n_hidden,layers):
        super(HNN, self).__init__()
        if layers == 2:
            self.net = nn.Sequential(
                nn.Linear(60, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden,1,bias=None)
            )
        if layers == 4:
            self.net = nn.Sequential(
                nn.Linear(60, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden,n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden,n_hidden),
                nn.Tanh(),
                nn.Linear(n_hidden,1,bias=None)
            )
        if layers == 6:
            self.net = nn.Sequential(
                nn.Linear(60, n_hidden),
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
                nn.Linear(n_hidden,1,bias=None)
            )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

    def derivative(self,data):
        F=self.forward(data)
        solenoidal_field=torch.zeros_like(data)
        dF=torch.autograd.grad(F.sum(),data,create_graph=True)[0]
        solenoidal_field=dF
        return solenoidal_field

    def sympletic(self,t,data):
        F=self.forward(data)
        solenoidal_field=torch.zeros_like(data)
        dF=torch.autograd.grad(F.sum(),data,create_graph=True)[0]
        solenoidal_field=torch.cat((dF[:,18:],-dF[:,:18]),dim=1)
        return solenoidal_field



def numer_deri(t,X):
    dt = t[1]-t[0]
    X_r = torch.cat((X[1:],X[:1]),dim=0)
    X_l = torch.cat((X[-1:],X[:-1]),dim=0)
    dX = (X_r-X_l)/(dt*2)
    return dX



class manybody(nn.Module):
    # n = 6, dim = 4*n
    def __init__(self,numbers):
        super(manybody, self).__init__()
        self.num = numbers

    def get_r(self,x,y):
        r = (x[:,0:1]-y[:,0:1])**2+(x[:,1:2]-y[:,1:2])**2
        output = torch.cat((-(x[:,0:1]-y[:,0:1])/r**1.5,-(x[:,1:2]-y[:,1:2])/r**1.5),dim=1)
        return output

    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        dx[:,:self.num*2] = x[:,self.num*2:]
        for i in range(self.num):
            for j in range(self.num):
                if j != i:
                    # print(dx[:, self.num * 2 + i * 2:self.num * 2 + i * 2 + 2].shape,
                    #       manybody.get_r(self, x[:, i * 2:i * 2 + 2], x[:, j * 2:j * 2 + 2]).shape)
                    dx[:,self.num*2+i*2:self.num*2+i*2+2] += manybody.get_r(self,x[:,i*2:i*2+2],x[:,j*2:j*2+2])

        return dx

def ana_deri(t,X):
    func = manybody(15)
    dX = func(0.0,X)
    return dX

true_y = np.load('./data/true_15_5_1000_symplectic_r7_a9.npy')[:160]
true_y = torch.from_numpy(true_y)

'''
learning
'''
sigma = args.sigma

out_iters = 0
Loss_list = []
while out_iters < 1:
    # break
    start = timeit.default_timer()
    # torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = HNN(args.neurons,args.layers)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    y = y.requires_grad_(True)
    # dy = numer_deri(t,y)[1:-1]
    dy=ana_deri(0.0,y)[1:-1]
    dq,dp = dy[:,:30],dy[:,30:]
    # print(dq.shape)
    i = 0
    max_iters = 20 # 19938,0.0;
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()],lr=learning_rate)
    while i < max_iters:
        # break
        vector = model.derivative(y)
        H_q,H_p = vector[1:-1,:30],vector[1:-1,30:]
        loss = torch.sum(torch.sqrt(torch.sum((H_p-dq)**2,dim=1)))\
               +torch.sum(torch.sqrt(torch.sum((H_q+dp)**2,dim=1)))
        print(out_iters,i, "loss=", loss.item())
        Loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)
    Loss = torch.tensor(Loss_list)
    Iter = torch.argmin(Loss).item()
    print('index of minimum:',torch.argmin(Loss))

    start = timeit.default_timer()
    # torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = HNN(args.neurons, args.layers)

    i = 0
    max_iters = Iter
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
    while i < max_iters:
        # break
        vector = model.derivative(y)
        H_q, H_p = vector[1:-1, :30], vector[1:-1, 30:]
        loss = torch.sum(torch.sqrt(torch.sum((H_p - dq) ** 2, dim=1))) \
               + torch.sum(torch.sqrt(torch.sum((H_q + dp) ** 2, dim=1)))
        print(out_iters, i, "loss=", loss.item())
        Loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss <= 1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)
    Loss = torch.tensor(Loss_list)
    print('index of minimum:', torch.argmin(Loss))





    y0 = true_y[0:1]
    y0.requires_grad_(True)
    t = torch.linspace(0, 5, 1000)
    pred_y=odeint(model.sympletic,y0,t,atol=1e-8,rtol=1e-8)[:,0,:].detach().numpy()
    np.save('./data/hnn_5_1000_tanh_{}_{}_{}'.format(sigma,args.layers,args.neurons),pred_y)
    plt.plot(true_y[:,0],true_y[:,1])
    plt.plot(pred_y[:,0],pred_y[:,1])
    out_iters += 1

# plt.show()