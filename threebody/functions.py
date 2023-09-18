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

def random_config(nu=0.02,min_radius=0.9,max_radius=1.2):
    '''This is not principled at all yet'''
    state=np.zeros((3,4))
    state[:,0]=1
    # p1=2*np.random.rand(2)-1
    # r=np.random.rand()*(max_radius-min_radius)+min_radius
    p1=np.array([0.5,0.5])
    r=min_radius
    print(p1)
    p1*=r/np.sqrt(np.sum((p1**2)))
    p2=rotate2d(p1,theta=2*np.pi/3)
    p3=rotate2d(p2,theta=2*np.pi/3)

    # # velocity that yields a circular orbit
    v1=rotate2d(p1,theta=np.pi/2)
    v1=v1/r**1.5
    v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories
    v2=rotate2d(v1,theta=2*np.pi/3)
    v3=rotate2d(v2,theta=2*np.pi/3)

    # make the circular orbits slightly chaotic
    v1*=1+nu*(2*np.random.rand(2)-1)
    v2*=1+nu*(2*np.random.rand(2)-1)
    v3*=1+nu*(2*np.random.rand(2)-1)
    p1+=nu*(2*np.random.rand(2))
    p2+=nu*(2*np.random.rand(2))
    p3+=nu*(2*np.random.rand(2))
    return np.concatenate((p1,p2,p3,v1,v2,v3))

def fix_config(nu=0.02,min_radius=0.9,max_radius=1.2):
    '''This is not principled at all yet'''
    # p1=2*np.random.rand(2)-1
    # r=np.random.rand()*(max_radius-min_radius)+min_radius
    p1=np.array([0.5,0.5])
    r=min_radius
    # print(p1)
    p1*=r/np.sqrt(np.sum((p1**2)))
    p2=rotate2d(p1,theta=2*np.pi/3)
    p3=rotate2d(p2,theta=2*np.pi/3)

    # # velocity that yields a circular orbit
    v1=rotate2d(p1,theta=np.pi/2)
    v1=v1/r**1.5
    v1=v1*np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2))  # scale factor to get circular trajectories
    v2=rotate2d(v1,theta=2*np.pi/3)
    v3=rotate2d(v2,theta=2*np.pi/3)

    return np.concatenate((p1,p2,p3,v1,v2,v3))

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

def plot0():
    n = 300
    # np.random.seed(7)
    y0 = torch.from_numpy(fix_config()).view([1,12])
    t = torch.linspace(0, 30, n)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
    A = rotate3d(-np.pi/50,np.pi/4)
    new_y,timeline = transform(A,true_y,t)
    # true_y = np.concatenate((true_y[:,:2],t.reshape(-1,1)),axis=1)
    # new_y = np.matmul(A,true_y.T).T
    print(true_y.shape,new_y.shape)
    # print(y0,true_y.shape)
    # plt.subplot(131)
    # plt.plot(true_y[:,0],true_y[:,1])
    co1 = 'blue'
    co2 = 'navy'
    co3 = 'royalblue'
    width = 3
    plt.plot(new_y[:,0],new_y[:,1],c=co1,ls='dashed',lw=width)
    plt.plot(new_y[:,2],new_y[:,3],c=co2,ls='dashed',lw=width)
    plt.plot(new_y[:,4],new_y[:,5],c=co3,ls='dashed',lw=width)
    plt.plot(timeline[:,0],timeline[:,1],c='black')
    plt.scatter(new_y[-1,0],new_y[-1,1],c=co1,marker='o',s=100)
    # plt.xlim(-2,2)
    # plt.ylim(-2,2)
    # plt.subplot(132)
    # plt.plot(true_y[:,0],true_y[:,1])
    # plt.subplot(133)
    # plt.plot(true_y[:,0],true_y[:,1])
    plt.show()

# plot0()


def generate():
    # random.seed = 369 for noise
    y0 = torch.from_numpy(fix_config()).view([1, 12])
    t = torch.linspace(0, 5, 100) # train
    train_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()
    t = torch.linspace(0,90,1800) # true
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()
    # np.save('./data/train_y_5_100',train_y)
    np.save('./data/true_y_90_1800',true_y)
# generate()

def subplot(data,timeline):
    co1 = colors[0]
    co2 = colors[1]
    co3 = colors[2]
    width = 1
    style = 'solid'
    plt.plot(timeline[:,0],timeline[:,1],c='grey')
    plt.plot(data[:,0],data[:,1],color=co1,ls=style,lw=width)
    plt.plot(data[:,2],data[:,3],color=co2,ls=style,lw=width)
    plt.plot(data[:,4],data[:,5],color=co3,ls=style,lw=width)
    plt.scatter(data[-1,0],data[-1,1],color=co1,marker='o',s=100)
    plt.scatter(data[-1,2], data[-1,3], color=co2, marker='o', s=100)
    plt.scatter(data[-1, 4], data[-1, 5], color=co3, marker='o', s=100)
    plt.xticks([])
    plt.yticks([])
    # plt.xlim(-2,2)
    # plt.ylim(-2,2)

def subsubplot(data):
    left,bottom,width,height = 0.2,0.6,0.25,0.25
    plt.axes([bottom,left,width,height])
    plt.plot(data[:,0],data[:,1],color=colors[0])
    # plt.xticks([])
    # plt.yticks([])

def plot():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 15
    labelsize= 15
    true = np.load('./data/true_y_90_1800.npy')[:1000]
    hnn = np.load('./data/hnn_90_1800_0.03.npy')[:1000]
    hnko = np.load('./data/HNKO_90_1800_0.03.npy')[:1000]
    edmd = np.load('./data/enc_edmd_90_1800_0.03.npy')[:1000]
    # node = np.load('./data/node_30_600.npy')
    # y0 = torch.from_numpy(fix_config()).view([1,12])
    t = np.linspace(0, 50, 1000)
    t_t = np.linspace(0, 50, 1000)
    A = rotate3d(-np.pi/50,np.pi*0.3)
    t_true,timeline = transform(A,true,t)
    t_hnn, timeline = transform(A, hnn, t_t)
    t_hnko,timeline = transform(A,hnko, t_t)
    t_edmd, timeline = transform(A,edmd, t_t)
    # t_node, timeline = transform(A,node, t_t)
    plt.subplot(321)
    subplot(t_true,timeline)
    plt.title('Truth',fontsize=fontsize)
    plt.subplot(322)
    subplot(t_hnn,timeline)
    plt.title('HNN',fontsize=fontsize)
    plt.subplot(323)
    subplot(t_edmd,timeline)
    plt.title('EDMD',fontsize=fontsize)
    plt.subplot(324)
    subplot(t_hnko,timeline)
    plt.title('HNKO',fontsize=fontsize)
    plt.subplot(325)
    plt.plot(t,state_error(true,hnn),label='HNN')
    plt.plot(t,state_error(true,edmd),label='EDMD')
    plt.plot(t,state_error(true,hnko),label='HNKO')
    plt.xticks([0,25,50],fontsize=labelsize)
    plt.yticks([0,8,16],fontsize=labelsize)
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel('State error',fontsize=fontsize)
    plt.subplot(326)
    plt.plot(t,hami(hnn),label='HNN')
    plt.plot(t,hami(edmd),label='EDMD')
    plt.plot(t,hami(hnko),label='HNKO')
    plt.plot(t,hami(true),color='black',label='Truth',ls=(0,(3,3)))
    plt.xticks([0,25,50],fontsize=labelsize)
    plt.yticks([-2,0,5],fontsize=labelsize)
    plt.legend(ncol=4,bbox_to_anchor=[0,-0.3],fontsize=10,frameon=False)
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel('Energy',fontsize=fontsize)

    plt.show()

def plot2():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    fontsize = 15
    labelsize = 15
    true = np.load('./data/true_y_90_1800.npy')[:1000]
    hnn = np.load('./data/hnn_90_1800_0.03.npy')[:1000]
    hnko = np.load('./data/HNKO_90_1800_0.03.npy')[:1000]
    edmd = np.load('./data/enc_edmd_90_1800_0.03.npy')[:1000]
    # node = np.load('./data/node_30_600.npy')
    # y0 = torch.from_numpy(fix_config()).view([1,12])
    t = np.linspace(0, 50, 1000)
    plt.plot(t, state_error(true, edmd), label='EDMD')
    plt.plot(t,state_error(true,edmd),label='EDMD')
    plt.plot(t,state_error(true,hnko),label='HNKO')
    plt.xticks([0,25,50],fontsize=labelsize)
    plt.yticks([0,0.5],fontsize=labelsize)
    # plt.xlabel('Time',fontsize=fontsize)
    # plt.ylabel('State error',fontsize=fontsize)
    plt.show()
# plot()