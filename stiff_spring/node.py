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
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=int, default=0.003)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
from functions import *
torch.manual_seed(2)

n = args.data_size
y0 = torch.tensor([[np.sqrt(2/1000),0.0]])
t = torch.linspace(0, 3, n)
true_y = odeint(spring(), y0, t, atol=1e-3, rtol=1e-3)
plt.scatter(true_y[:,:,0],true_y[:,:,1])

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size-args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100,2),
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
    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=0.0)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    # batch_y0, batch_t, batch_y = get_batch()
    # print(batch_y0.shape,batch_y.shape)
    # pred_y = odeint(func, batch_y0, batch_t)
    # print(pred_y.shape)
    b = args.batch_time*(t[1]-t[0])/2
    for itr in range(1, args.niters + 1):
        # break

        batch_y0, batch_t, batch_y = get_batch()
        batch_t += torch.Tensor(1).uniform_(-b,b)
        pred_y = odeint(func, batch_y0, batch_t)
        loss=torch.mean(torch.abs(pred_y[-1,:]-batch_y[-1,:]))
        loss.requires_grad_(True)
        # if loss < 1.00:
        #     optimizer=optim.Adam(func.parameters(),lr=0.001,weight_decay=0.0)
        # else:
        #     optimizer=optim.Adam(func.parameters(),lr=args.lr,weight_decay=0.0)
        if loss < 0.01:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        print(itr,loss)
        end = time.time()
    endt = timeit.default_timer()
    print(endt-start)
    pred_t=torch.linspace(0,6,n)
    pred_y=odeint(func,y0,pred_t,atol=1e-6,rtol=1e-6).detach().numpy()
    plt.scatter(pred_y[:,:,0],pred_y[:,:,1])
    plt.show()
# torch.save(func.state_dict(),'train.pkl')