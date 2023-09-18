import numpy as np
import torch

from functions import *


class HNN(nn.Module):

    def __init__(self,n_hidden):
        super(HNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(144, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
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
        solenoidal_field=torch.cat((dF[:,72:],-dF[:,:72]),dim=1)
        return solenoidal_field



def numer_deri(t,X):
    dt = t[1]-t[0]
    X_r = torch.cat((X[1:],X[:1]),dim=0)
    X_l = torch.cat((X[-1:],X[:-1]),dim=0)
    dX = (X_r-X_l)/(dt*2)
    return dX




class manybody3D(nn.Module):
    # dim = 6*n
    def __init__(self,numbers):
        super(manybody3D, self).__init__()
        self.num = numbers

    def get_r(self,x,y):
        r = (x[:,0:1]-y[:,0:1])**2+(x[:,1:2]-y[:,1:2])**2+(x[:,2:3]-y[:,2:3])**2
        return torch.cat((-(x[:,0:1]-y[:,0:1])/r**1.5,-(x[:,1:2]-y[:,1:2])/r**1.5,-(x[:,2:3]-y[:,2:3])/r**1.5),dim=1)

    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        dx[:,:self.num*3] = x[:,self.num*3:]
        for i in range(self.num):
            for j in range(self.num):
                if j != i:
                    dx[:,self.num*3+i*3:self.num*3+i*3+3] += manybody3D.get_r(self,x[:,i*3:i*3+3],x[:,j*3:j*3+3])

        return dx

def ana_deri(t,X):
    func = manybody3D(24)
    dX = func(0.0,X)
    return dX

true_y = np.load('./data/3d_24body_full.npy')
true_y = torch.from_numpy(true_y)

'''
learning
'''
sigma = 0.03
H1 = 64
out_iters = 0
Loss_list = []

while out_iters < 1:
    # break
    start = timeit.default_timer()
    # torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = HNN(H1)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    y = y.requires_grad_(True)
    # dy = numer_deri(t,y)[1:-1]
    dy=ana_deri(0.0,y)
    dq,dp = dy[:,:72],dy[:,72:]
    # print(dq.shape)
    i = 0
    max_iters = 10000
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()],lr=learning_rate)
    while i < max_iters:
        # break
        s = torch.from_numpy(np.random.choice(np.arange(1800,dtype=np.int64), 30, replace=False))
        batch_y = y[s]
        batch_dq,batch_dp = dq[s],dp[s]
        vector = model.derivative(batch_y)
        H_q,H_p = vector[:,:72],vector[:,72:]
        loss = torch.sum(torch.sqrt(torch.sum((H_p-batch_dq)**2,dim=1)))\
               +torch.sum(torch.sqrt(torch.sum((H_q+batch_dp)**2,dim=1)))
        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        Loss_list.append(loss.item())
        if loss.item() == min(Loss_list):
            best_model = model.sympletic
        i += 1

    Loss = torch.tensor(Loss_list)
    print('Best results: ({},{})'.format(torch.argmin(Loss),torch.min(Loss)))
    y0 = true_y[0:1]
    y0.requires_grad_(True)
    t = torch.linspace(0, 1, 1800)
    # test_y0 = torch.from_numpy(fix_config()).view([1,12])
    # test_y0.requires_grad_(True)
    # true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
    pred_y=odeint(best_model,y0,t,atol=1e-8,rtol=1e-8)[:,0,:].detach().numpy()
    np.save('./data/hnn_{}_3_{}'.format(sigma,H1),pred_y)
    plt.plot(true_y[:,0],true_y[:,1])
    plt.plot(pred_y[:,0],pred_y[:,1])
    stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)

    out_iters += 1

plt.show()