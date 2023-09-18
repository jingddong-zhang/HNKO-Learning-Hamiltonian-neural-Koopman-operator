import numpy as np
import torch

from functions import *


class HNN(nn.Module):

    def __init__(self):
        super(HNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(12, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200,200),
            nn.Tanh(),
            nn.Linear(200,200),
            nn.Tanh(),
            nn.Linear(200,1,bias=None)
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
        solenoidal_field=torch.cat((dF[:,6:],-dF[:,:6]),dim=1)
        return solenoidal_field



def numer_deri(t,X):
    dt = t[1]-t[0]
    X_r = torch.cat((X[1:],X[:1]),dim=0)
    X_l = torch.cat((X[-1:],X[:-1]),dim=0)
    dX = (X_r-X_l)/(dt*2)
    return dX


def ana_deri(t,X):
    func = threebody()
    dX = func(0.0,X)
    return dX

y0=torch.from_numpy(fix_config()).view([1,12])
t = torch.linspace(0, 5, 100)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]

# plt.plot(true_y[:,0],true_y[:,1])

'''
learning
'''
sigma = 0.03


out_iters = 0
while out_iters < 1:
    # break
    start = timeit.default_timer()
    # torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = HNN()
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    y = y.requires_grad_(True)
    # dy = numer_deri(t,y)[1:-1]
    dy=ana_deri(t,y)[1:-1]
    dq,dp = dy[:,:6],dy[:,6:]
    # print(dq.shape)
    i = 0
    max_iters = 10000
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()],lr=learning_rate)
    while i < max_iters:
        # break
        vector = model.derivative(y)
        H_q,H_p = vector[1:-1,:6],vector[1:-1,6:]
        loss = torch.sum(torch.sqrt(torch.sum((H_p-dq)**2,dim=1)))\
               +torch.sum(torch.sqrt(torch.sum((H_q+dp)**2,dim=1)))
        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)
    y0.requires_grad_(True)
    # test_y0 = torch.from_numpy(fix_config()).view([1,12])
    # test_y0.requires_grad_(True)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
    pred_y=odeint(model.sympletic,y0,torch.linspace(0, 90, 1800),atol=1e-8,rtol=1e-8)[:,0,:].detach().numpy()
    np.save('./data/hnn_nu_90_1800_0.03',pred_y)
    plt.plot(true_y[:,0],true_y[:,1])
    plt.plot(pred_y[:,0],pred_y[:,1])
    # plt.plot(true_y[:,2],true_y[:,3])
    # plt.plot(pred_y[:,2],pred_y[:,3])
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1

plt.show()