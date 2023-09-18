import numpy as np
import torch
import argparse
from functions import *
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=int, default=0.003)
args = parser.parse_args()

class HNN(nn.Module):

    def __init__(self):
        super(HNN, self).__init__()
        torch.manual_seed(2)
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
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
        solenoidal_field=torch.cat((dF[:,1:],-dF[:,:1]),dim=1)
        return solenoidal_field

def get_batch(true_y):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size-args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def numer_deri(t,X):
    dt = t[1]-t[0]
    X_r = torch.cat((X[1:],X[:1]),dim=0)
    X_l = torch.cat((X[-1:],X[:-1]),dim=0)
    dX = (X_r-X_l)/(dt*2)
    return dX


def ana_deri(t,X):
    # func = spring()
    dX = func(0.0,X)
    return dX


y0 = torch.tensor([[1.0,0.0]])
t = torch.linspace(0, 10, 100)
# y = odeint(pendulum(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
stiff_list = torch.linspace(1.0,1e5*1.2,40)
a = stiff_list[30]
k = torch.pow(a,1/2)
m = torch.pow(a,1/2)
func = spring(k,m)
print(k,m)
true_y = odeint(func, y0, t, atol=1e-8, rtol=1e-8)[:,0,:]

# n = 200
# y0 = torch.tensor([[1.0,0.0]])
# t = torch.linspace(0, 20, n)
# true_y = odeint(spring(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]

# true_y = np.load('./data/train_y_6_100.npy')
# t = torch.linspace(0,6,100)
# true_y = torch.from_numpy(true_y)
# y0 = true_y[0:1]
# plt.plot(true_y[:,0],true_y[:,1])

def hnn():
    train_data = np.load('./data/train_y_3_100.npy')
    y0 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0, 10, 100)
    out_iters = 0
    pred_data = np.zeros([40,300,2]) # save prediction data
    while out_iters < 40:
        # break
        start = timeit.default_timer()
        # torch.manual_seed(69)  # lift=0,q=6,seed=69
        model = HNN()
        y = torch.from_numpy(train_data[out_iters,:,:2])
        np.random.seed(369)
        y += torch.from_numpy(np.random.normal(0,0.03,y.shape))
        y = y.requires_grad_(True)
        dy = torch.from_numpy(train_data[out_iters,:,2:])
        dq, dp = dy[:, :1], dy[:, 1:]
        i = 0
        max_iters = 10000
        learning_rate = 0.001
        optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
        while i < max_iters:
            # break
            vector = model.derivative(y)
            H_q, H_p = vector[:, :1], vector[:, 1:]
            loss = torch.sum(torch.sqrt(torch.sum((H_p - dq) ** 2, dim=1))) \
                   + torch.sum(torch.sqrt(torch.sum((H_q + dp) ** 2, dim=1)))
            print(out_iters, i, "loss=", loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if loss <= 1e-4:
                break
            i += 1
        stop = timeit.default_timer()
        print('\n')
        print("Total time: ", stop - start)
        y0.requires_grad_(True)
        pred_t = torch.linspace(0,9,300)
        pred_y = odeint(model.sympletic, y0, pred_t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()
        pred_data[out_iters,:]=pred_y
        out_iters += 1
        # true_y = y.detach().numpy()
        # plt.plot(true_y[:,0],true_y[:,1])
        # plt.plot(pred_y[:, 0], pred_y[:, 1])
        # plt.show()
    np.save('./data/hnn_0.03_9_300',pred_data)
hnn()

'''
learning
'''
sigma = 0.0
out_iters = 0
while out_iters < 1:
    break
    start = timeit.default_timer()
    # torch.manual_seed(69)  # lift=0,q=6,seed=69
    # np.random.seed(369)
    model = HNN()
    y = true_y
    y = y.requires_grad_(True)
    batch_t = t
    # dy = numer_deri(t,y)[1:-1]
    # dy=ana_deri(t,y)[1:-1]
    # dq,dp = dy[:,:1],dy[:,1:]
    # print(dq.shape)
    i = 0
    max_iters = 2000
    learning_rate = 0.001
    optimizer = torch.optim.Adam([i for i in model.parameters()],lr=learning_rate)
    while i < max_iters:
        # break
        # batch_y0, batch_t, batch_y = get_batch(y)
        dy=ana_deri(batch_t,y)
        dq,dp=dy[:,:1],dy[:,1:]
        vector = model.derivative(y)
        H_q,H_p = vector[:,:1],vector[:,1:]
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
    # test_y0 = torch.tensor([[1.0,0.0]])
    # test_y0.requires_grad_(True)
    true_y = odeint(func, y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
    pred_y=odeint(model.sympletic,y0,t,atol=1e-8,rtol=1e-8)[:,0,:].detach().numpy()
    # np.save('./data/hnn_0.03',pred_y)
    plt.subplot(121)
    plt.plot(true_y[:,0],true_y[:,1])
    plt.subplot(122)
    plt.plot(pred_y[:,0],pred_y[:,1])
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1

plt.show()