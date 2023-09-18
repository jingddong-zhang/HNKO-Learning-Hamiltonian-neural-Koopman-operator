import argparse
import time
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100) # 300
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr', type=int, default=0.01)
parser.add_argument('--dim', type=int, default=1024)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
from functions import *
torch.manual_seed(369)

data = np.load('./data/samples_{}_501.npy'.format(args.dim),allow_pickle=True).item()
t = data['t']
t = torch.from_numpy(t)
true_y = data['u_x']
dim = true_y.shape[1]
sigma = 0.03
np.random.seed(369)
noise = np.random.normal(0,sigma,true_y.shape)
train_y = true_y + noise

train_y = torch.from_numpy(train_y)[:args.data_size]
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size-args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = train_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([train_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


class ODEFunc(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    start = timeit.default_timer()
    ii = 0
    func = ODEFunc(dim,int(1.1*dim),dim)
    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=0.0)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    batch_y0, batch_t, batch_y = get_batch()

    for itr in range(1, args.niters + 1):
        # break

        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t,method='rk4',options=dict(step_size=0.01))
        loss=torch.mean(torch.abs(pred_y[-1,:]-batch_y[-1,:]))
        loss.requires_grad_(True)
        if loss < 0.01:
            break
        if itr%300 == 0:
            torch.save(func.state_dict(), './data/node_{}_{}.pkl'.format(args.dim,itr))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        print(itr,loss)
        end = time.time()
    endt = timeit.default_timer()
    print(endt-start)

    torch.save(func.state_dict(),'./data/node_{}.pkl'.format(args.dim))

    pred_t = t
    y0 = torch.from_numpy(true_y[0]).view([1,-1])
    pred_y=odeint(func,y0,pred_t,atol=1e-6,rtol=1e-6).detach().numpy()[:,0,:]
    error = np.sum((true_y - pred_y)**2, axis=1)
    print('NODE MSE:',error[400:].mean())