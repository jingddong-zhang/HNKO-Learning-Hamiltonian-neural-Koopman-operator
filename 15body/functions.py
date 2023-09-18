import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import cm
import matplotlib as mpl
from torchdiffeq import odeint
import geotorch
torch.set_default_dtype(torch.float64)


colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 236/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
]
class threebody(nn.Module):
    dim = 12
    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
        dx[:, 0] = a1
        dx[:, 1] = b1
        dx[:, 2] = a2
        dx[:, 3] = b2
        dx[:, 4] = a3
        dx[:, 5] = b3
        r12 = ((x1-x2)**2+(y1-y2)**2)**1.5
        r13 = ((x1-x3)**2+(y1-y3)**2)**1.5
        r23 = ((x2-x3)**2+(y2-y3)**2)**1.5
        dx[:, 6] = -(x1-x2)/r12-(x1-x3)/r13
        dx[:, 7] = -(y1-y2)/r12-(y1-y3)/r13
        dx[:, 8] = -(x2-x1)/r12-(x2-x3)/r23
        dx[:, 9] = -(y2-y1)/r12-(y2-y3)/r23
        dx[:, 10] = -(x3-x1)/r13-(x3-x2)/r23
        dx[:, 11] = -(y3-y1)/r13-(y3-y2)/r23
        return dx

class nbody(nn.Module):
    # n = 6, dim = 4*n
    dim = 24
    def get_r(self,x,y):
        r = (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
        return torch.cat((-(x[:,0:1]-y[:,0:1])/r**1.5,-(x[:,1:2]-y[:,1:2])/r**1.5),dim=1)
    def get_r_y(self,x,y):
        r = (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
        return -(x[:,1]-y[:,1])/r**1.5

    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
        # a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
        dx[:,:12] = x[:,12:]
        r12 = ((x1-x2)**2+(y1-y2)**2)**1.5
        r13 = ((x1-x3)**2+(y1-y3)**2)**1.5
        r23 = ((x2-x3)**2+(y2-y3)**2)**1.5
        dx[:, 12:14] = nbody.get_r(self,x[:,0:2],x[:,2:4])+nbody.get_r(self,x[:,0:2],x[:,4:6])+nbody.get_r(self,x[:,0:2],x[:,6:8])+nbody.get_r(self,x[:,0:2],x[:,8:10])+nbody.get_r(self,x[:,0:2],x[:,10:12])
        dx[:, 14:16] = nbody.get_r(self,x[:,2:4],x[:,0:2])+nbody.get_r(self,x[:,2:4],x[:,4:6])+nbody.get_r(self,x[:,2:4],x[:,6:8])+nbody.get_r(self,x[:,2:4],x[:,8:10])+nbody.get_r(self,x[:,2:4],x[:,10:12])
        dx[:, 16:18] = nbody.get_r(self,x[:,4:6],x[:,0:2])+nbody.get_r(self,x[:,4:6],x[:,2:4])+nbody.get_r(self,x[:,4:6],x[:,6:8])+nbody.get_r(self,x[:,4:6],x[:,8:10])+nbody.get_r(self,x[:,4:6],x[:,10:12])
        dx[:, 18:20] = nbody.get_r(self,x[:,6:8],x[:,0:2])+nbody.get_r(self,x[:,6:8],x[:,2:4])+nbody.get_r(self,x[:,6:8],x[:,4:6])+nbody.get_r(self,x[:,6:8],x[:,8:10])+nbody.get_r(self,x[:,6:8],x[:,10:12])
        dx[:, 20:22] = nbody.get_r(self,x[:,8:10],x[:,0:2])+nbody.get_r(self,x[:,8:10],x[:,2:4])+nbody.get_r(self,x[:,8:10],x[:,4:6])+nbody.get_r(self,x[:,8:10],x[:,6:8])+nbody.get_r(self,x[:,8:10],x[:,10:12])
        dx[:, 22:24] = nbody.get_r(self,x[:,10:12],x[:,0:2])+nbody.get_r(self,x[:,10:12],x[:,2:4])+nbody.get_r(self,x[:,10:12],x[:,4:6])+nbody.get_r(self,x[:,10:12],x[:,6:8])+nbody.get_r(self,x[:,10:12],x[:,8:10])
        return dx


class manybody(nn.Module):
    # n = 6, dim = 4*n
    def __init__(self,numbers):
        super(manybody, self).__init__()
        self.num = numbers

    def get_r(self,x,y):
        r = (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
        return torch.cat((-(x[:,0:1]-y[:,0:1])/r**1.5,-(x[:,1:2]-y[:,1:2])/r**1.5),dim=1)
    def get_r_y(self,x,y):
        r = (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
        return -(x[:,1]-y[:,1])/r**1.5

    def forward(self, t, x):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(x)
        # x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
        # # a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
        # dx[:,:12] = x[:,12:]
        # r12 = ((x1-x2)**2+(y1-y2)**2)**1.5
        # r13 = ((x1-x3)**2+(y1-y3)**2)**1.5
        # r23 = ((x2-x3)**2+(y2-y3)**2)**1.5
        # dx[:, 12:14] = nbody.get_r(self,x[:,0:2],x[:,2:4])+nbody.get_r(self,x[:,0:2],x[:,4:6])+nbody.get_r(self,x[:,0:2],x[:,6:8])+nbody.get_r(self,x[:,0:2],x[:,8:10])+nbody.get_r(self,x[:,0:2],x[:,10:12])
        # dx[:, 14:16] = nbody.get_r(self,x[:,2:4],x[:,0:2])+nbody.get_r(self,x[:,2:4],x[:,4:6])+nbody.get_r(self,x[:,2:4],x[:,6:8])+nbody.get_r(self,x[:,2:4],x[:,8:10])+nbody.get_r(self,x[:,2:4],x[:,10:12])
        # dx[:, 16:18] = nbody.get_r(self,x[:,4:6],x[:,0:2])+nbody.get_r(self,x[:,4:6],x[:,2:4])+nbody.get_r(self,x[:,4:6],x[:,6:8])+nbody.get_r(self,x[:,4:6],x[:,8:10])+nbody.get_r(self,x[:,4:6],x[:,10:12])
        # dx[:, 18:20] = nbody.get_r(self,x[:,6:8],x[:,0:2])+nbody.get_r(self,x[:,6:8],x[:,2:4])+nbody.get_r(self,x[:,6:8],x[:,4:6])+nbody.get_r(self,x[:,6:8],x[:,8:10])+nbody.get_r(self,x[:,6:8],x[:,10:12])
        # dx[:, 20:22] = nbody.get_r(self,x[:,8:10],x[:,0:2])+nbody.get_r(self,x[:,8:10],x[:,2:4])+nbody.get_r(self,x[:,8:10],x[:,4:6])+nbody.get_r(self,x[:,8:10],x[:,6:8])+nbody.get_r(self,x[:,8:10],x[:,10:12])
        # dx[:, 22:24] = nbody.get_r(self,x[:,10:12],x[:,0:2])+nbody.get_r(self,x[:,10:12],x[:,2:4])+nbody.get_r(self,x[:,10:12],x[:,4:6])+nbody.get_r(self,x[:,10:12],x[:,6:8])+nbody.get_r(self,x[:,10:12],x[:,8:10])

        dx[:,:self.num*2] = x[:,self.num*2:]
        for i in range(self.num):
            for j in range(self.num):
                if j != i:
                    dx[:,self.num*2+i*2:self.num*2+i*2+2] += manybody.get_r(self,x[:,i*2:i*2+2],x[:,j*2:j*2+2])

        return dx


def hami(x):
    x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
    h = (a1**2+b1**2+a2**2+b2**2+a3**2+b3**2)/2-1/(np.sqrt((x1-x2)**2+(y1-y2)**2))\
            -1/(np.sqrt((x1-x3)**2+(y1-y3)**2))-1/(np.sqrt((x3-x2)**2+(y3-y2)**2))
    return h

def state_error(true,pred):
    err = np.sqrt(np.sum((true-pred)**2,axis=1))
    return err

def rotate2d(p, theta):
  c, s = np.cos(theta), np.sin(theta)
  R = np.array([[c, -s],[s, c]])
  return (R @ p.reshape(2,1)).squeeze()




# def fix_config(nu=0.02,min_radius=0.9,max_radius=1.2):
#     '''This is not principled at all yet'''
#     # p1=2*np.random.rand(2)-1
#     # r=np.random.rand()*(max_radius-min_radius)+min_radius
#     p1=np.array([0.5,0.5])
#     r=min_radius
#     # print(p1)
#     p1*=r/np.sqrt(np.sum((p1**2)))
#     p2=rotate2d(p1,theta=2*np.pi/3)
#     p3=rotate2d(p2,theta=2*np.pi/3)
#
#     # # velocity that yields a circular orbit
#     v1=rotate2d(p1,theta=np.pi/2)
#     v1=v1/r**1.5
#     v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories
#     v2=rotate2d(v1,theta=2*np.pi/3)
#     v3=rotate2d(v2,theta=2*np.pi/3)
#
#     return np.concatenate((p1,p2,p3,v1,v2,v3))

def fix_config6(num=6,min_radius=0.9,max_radius=1.2):
    '''This is not principled at all yet'''
    # p1=2*np.random.rand(2)-1
    # r=np.random.rand()*(max_radius-min_radius)+min_radius
    p1=np.array([0.5,0.5])
    r=min_radius
    # print(p1)
    p1*=r/np.sqrt(np.sum((p1**2)))
    p2=rotate2d(p1,theta=2*np.pi/6)
    p3=rotate2d(p2,theta=2*np.pi/6)
    p4=rotate2d(p3,theta=2*np.pi/6)
    p5=rotate2d(p4,theta=2*np.pi/6)
    p6=rotate2d(p5,theta=2*np.pi/6)


    # # velocity that yields a circular orbit
    v1=rotate2d(p1,theta=np.pi/2)
    v1=v1/r**1.5
    v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories
    v2=rotate2d(v1,theta=2*np.pi/6)
    v3=rotate2d(v2,theta=2*np.pi/6)
    v4=rotate2d(v3,theta=2*np.pi/6)
    v5=rotate2d(v4,theta=2*np.pi/6)
    v6=rotate2d(v5,theta=2*np.pi/6)
    output = np.concatenate((p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6))
    return output


def fix_config(num=6,min_radius=0.9,max_radius=1.2):
    '''This is not principled at all yet'''
    # p1=2*np.random.rand(2)-1
    # r=np.random.rand()*(max_radius-min_radius)+min_radius
    output = np.zeros(num*4)
    p1=np.array([0.5,0.5])
    r=min_radius
    # print(p1)
    p1*=r/np.sqrt(np.sum((p1**2)))
    # p2=rotate2d(p1,theta=2*np.pi/6)
    # p3=rotate2d(p2,theta=2*np.pi/6)
    # p4=rotate2d(p3,theta=2*np.pi/6)
    # p5=rotate2d(p4,theta=2*np.pi/6)
    # p6=rotate2d(p5,theta=2*np.pi/6)

    output[0:2] = p1
    for i in range(1,num):
        p = output[i*2-2:i*2]
        output[i*2:i*2+2] = rotate2d(p,theta=2*np.pi/num)

    # # velocity that yields a circular orbit
    v1=rotate2d(p1,theta=np.pi/2)
    v1=v1/r**1.5
    v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories
    # v2=rotate2d(v1,theta=2*np.pi/6)
    # v3=rotate2d(v2,theta=2*np.pi/6)
    # v4=rotate2d(v3,theta=2*np.pi/6)
    # v5=rotate2d(v4,theta=2*np.pi/6)
    # v6=rotate2d(v5,theta=2*np.pi/6)
    output[num*2:num*2+2] = v1
    for i in range(1,num):
        v = output[num*2+i*2-2:num*2+i*2]
        output[num*2+i*2:num*2+i*2+2] = rotate2d(v,theta=2*np.pi/num)
    return output

# start = timeit.default_timer()
# num = 25
# y0 = torch.from_numpy(fix_config(num)).view([1,4*num])
# t = torch.linspace(0, 5, 100)
# func = manybody(num)
# true_y = odeint(func, y0, t, atol=1e-10, rtol=1e-10)[:,0,:].detach().numpy()
# end = timeit.default_timer()
# plt.plot(t,true_y[:,0])
# print(true_y.shape,end-start)
# np.save('./data/true_25_5_100',true_y)
# plt.show()

def sphere_coord(theta,phi):
    r = 1.0
    return np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])

def rotate3d(theta,phi):
    v1 = sphere_coord(theta,phi)
    v2 = sphere_coord(np.pi/2,np.pi*3/4)
    u = np.array([1.0,0.0,0.0])
    v3 = u-np.dot(u,v1)*v1-np.dot(u,v2)*v2
    v3 = v3/np.linalg.norm(v3,2)
    A = np.matrix([v3,v2,v1]).T
    return A


def transform(A,trajectory,t):
    # shape of trajectory: (n,12)
    trans_data = np.zeros([len(trajectory),6]) # 3-body time trajectory projected after rotation in state-time space
    t = t.reshape(-1,1) # convert to column vector
    for i in range(3):
        time_aug = np.concatenate((trajectory[:,2*i:2*i+2], t), axis=1)
        trans = np.matmul(A, time_aug.T).T
        trans_data[:,2*i:2*i+2] = trans[:,:2]
    timeline = np.concatenate((np.zeros([len(trajectory),2]),t),axis=1)
    timeline = np.matmul(A, timeline.T).T
    return trans_data,timeline[:,:2]

def generate():
    # random.seed = 369 for noise
    y0 = torch.from_numpy(fix_config6()).view([1, 12])
    t = torch.linspace(0, 5, 100) # train
    train_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()
    t = torch.linspace(0,90,1800) # true
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()
    # np.save('./data/train_y_5_100',train_y)
    np.save('./data/true_y_90_1800',true_y)
# generate()

