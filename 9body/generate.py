import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import cm
import matplotlib as mpl
from torchdiffeq import odeint_adjoint as odeint
import argparse
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--num', type=int, default=15)
parser.add_argument('--tol', type=float, default=1e-8)
args = parser.parse_args()


class manybody(nn.Module):
    # n = 6, dim = 4*n
    def __init__(self,numbers):
        super(manybody, self).__init__()
        self.num = numbers

    def get_r(self,x,y):
        r = (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
        return torch.cat((-(x[:,0:1]-y[:,0:1])/r**1.5,-(x[:,1:2]-y[:,1:2])/r**1.5),dim=1)

    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        dx[:,:self.num*2] = x[:,self.num*2:]
        for i in range(self.num):
            for j in range(self.num):
                if j != i:
                    dx[:,self.num*2+i*2:self.num*2+i*2+2] += manybody.get_r(self,x[:,i*2:i*2+2],x[:,j*2:j*2+2])

        return dx


def rotate2d(p, theta):
  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c, -s],[s, c]])
  return (R @ p.reshape(2,1)).squeeze()

def fix_config(num=6,min_radius=0.9,max_radius=1.2):
    '''This is not principled at all yet'''
    # p1=2*np.random.rand(2)-1
    # r=np.random.rand()*(max_radius-min_radius)+min_radius
    output = np.zeros(num*4)
    p1=np.array([0.5,0.5])
    r=min_radius
    # print(p1)
    p1*=r/np.sqrt(np.sum((p1**2)))

    output[0:2] = p1
    for i in range(1,num):
        p = output[i*2-2:i*2]
        output[i*2:i*2+2] = rotate2d(p,theta=2*np.pi/num)

    # # velocity that yields a circular orbit
    v1=rotate2d(p1,theta=np.pi/2)
    v1=v1/r**1.5
    v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories

    output[num*2:num*2+2] = v1
    for i in range(1,num):
        v = output[num*2+i*2-2:num*2+i*2]
        output[num*2+i*2:num*2+i*2+2] = rotate2d(v,theta=2*np.pi/num)
    return output

start = timeit.default_timer()
num = args.num

y0 = torch.from_numpy(fix_config(num)).view([1,4*num])
t = torch.linspace(0, 5, 100)
func = manybody(num)
# true_y = odeint(func, y0, t, atol=args.tol, rtol=args.tol)[:,0,:].detach().numpy()
true_y = odeint(func, y0, t, method='rk4',options=dict(step_size=1e-6))[:,0,:].detach().numpy()

# true_y = np.load('./data/true_15_5_100.npy')
end = timeit.default_timer()
plt.plot(t,true_y[:,0])
print(true_y.shape,end-start)
np.save('./data/true_{}_5_100'.format(num),true_y)
plt.show()

