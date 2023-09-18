import numpy as np
import matplotlib.pyplot as plt


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
'''
functions
'''
def t_energy(x):
    x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
    h = (a1**2+b1**2+a2**2+b2**2+a3**2+b3**2)/2-1/(np.sqrt((x1-x2)**2+(y1-y2)**2))\
            -1/(np.sqrt((x1-x3)**2+(y1-y3)**2))-1/(np.sqrt((x3-x2)**2+(y3-y2)**2))
    return np.log(np.abs(h)) if h[0]>0 else -np.log10(np.abs(h))

def k_energy(x):
    x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
    h = (a1**2+b1**2+a2**2+b2**2+a3**2+b3**2)/2
    return np.log(np.abs(h)) if h[0]>0 else -np.log10(np.abs(h))

def p_energy(x):
    x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
    h = -1/(np.sqrt((x1-x2)**2+(y1-y2)**2))\
            -1/(np.sqrt((x1-x3)**2+(y1-y3)**2))-1/(np.sqrt((x3-x2)**2+(y3-y2)**2))
    return np.log(np.abs(h)) if h[0]>0 else -np.log10(np.abs(h))

def state_error(true,pred):
    err = np.sqrt(np.sum((true-pred)**2,axis=1))
    return err

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

'''
plot
'''
fontsize = 15
labelsize= 15


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

def planr_plot(data):
    co1 = colors[0]
    co2 = colors[1]
    co3 = colors[2]
    width = 1
    style = 'solid'
    plt.plot(data[:,0],data[:,1],color=co1,ls=style,lw=width)
    plt.plot(data[:,2],data[:,3],color=co2,ls=style,lw=width)
    plt.plot(data[:,4],data[:,5],color=co3,ls=style,lw=width)
    plt.scatter(data[-1,0],data[-1,1],color=co1,marker='o',s=100)
    plt.scatter(data[-1,2], data[-1,3], color=co2, marker='o', s=100)
    plt.scatter(data[-1, 4], data[-1, 5], color=co3, marker='o', s=100)
    plt.xticks([])
    plt.yticks([])

def energy_plot(X,Y,t):
    co = colors[2]
    plt.plot(t,k_energy(Y),label='Kinetic',color=co,ls='--')
    plt.plot(t,p_energy(Y),label='Total',color=co,ls='dotted')
    plt.plot(t,t_energy(Y),label='Potential',color=co,ls='-')
    plt.plot(t,k_energy(X),color='black',ls='--')
    plt.plot(t,p_energy(X),color='black',ls='dotted')
    plt.plot(t,t_energy(X),color='black',ls='-')

    plt.ylabel('Energy',fontsize=fontsize)
    plt.ylim(-0.35,0.1)
    # plt.legend(frameon=False, loc=5)

def case_study():
    true = np.load('./data/true_y_90_1800.npy')[:1000]
    case1 = np.load('./data/hnko_20_15_90_1800_0.03.npy')[:1000]
    case2 = np.load('./data/hnko_25_15_90_1800_0.03.npy')[:1000]
    case3 = np.load('./data/hnko_90_1800_0.03.npy')[:1000]
    case4 = np.load('./data/hnko_31_15_90_1800_0.03.npy')[:1000]
    t = np.linspace(0,50,1000)
    # plt.plot(t,state_error(true,case1))
    # plt.plot(t,state_error(true,case2))
    plt.plot(t,state_error(true,case3))
    plt.plot(t,state_error(true,case4))
    plt.show()

case_study()

def plot():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'

    '''
    data size: (1800,12) 
    1800: time interval on [0,90], 1800 uniform samples
    12:   state q_x1,q_y1,q_x2,q_y2,q_x3,q_y3,p_x1,p_y1,p_x2,p_y2,p_x3,p_y3
    here we choose time intrval on [0,50] for plot, that is, data[:1000,:]
    '''
    true = np.load('./data/true_y_90_1800.npy')[:1000]
    hnn = np.load('./data/hnn_90_1800_0.03.npy')[:1000]
    hnko = np.load('./data/HNKO_90_1800_0.03.npy')[:1000]
    edmd = np.load('./data/enc_edmd_90_1800_0.03.npy')[:1000]
    edmd_he = np.load('./data/he_edmd_90_1800_0.03.npy')[:1000]
    node = np.load('./data/node_90_1800_0.03.npy')[:1000]
    hnn_nu = np.load('./data/hnn_nu_90_1800_0.03.npy')[:1000]
    '''
    3D space-time projection to 2D plane 
    '''
    t = np.linspace(0, 50, 1000)
    A = rotate3d(-np.pi/50,np.pi*0.3)
    t_true,timeline = transform(A,true,t)
    t_hnn, timeline = transform(A, hnn, t)
    t_hnko,timeline = transform(A,hnko, t)
    t_edmd, timeline = transform(A,edmd, t)
    t_node, timeline = transform(A,node, t)
    t_edmd_he, timeline = transform(A,edmd_he, t)
    t_hnn_nu, timeline = transform(A,hnn_nu, t)
    '''
    plot
    '''
    plt.subplot(631)
    subplot(t_hnn,timeline)
    plt.ylabel('HNN(+AN)',fontsize=fontsize)
    plt.subplot(634)
    subplot(t_hnn_nu, timeline)
    plt.ylabel('HNN(+NU)',fontsize=fontsize)
    plt.subplot(637)
    subplot(t_edmd, timeline)
    plt.ylabel('EDMD(+AE)',fontsize=fontsize)
    plt.subplot(6,3,10)
    subplot(t_edmd_he,timeline)
    plt.ylabel('EDMD(+HE)',fontsize=fontsize)
    plt.subplot(6,3,13)
    subplot(t_node, timeline)
    plt.ylabel('NODE',fontsize=fontsize)
    plt.subplot(6,3,16)
    subplot(t_hnko,timeline)
    plt.ylabel('HNKO',fontsize=fontsize)

    plt.subplot(6,3,2)
    planr_plot(hnn)
    plt.subplot(6,3,5)
    planr_plot(hnn_nu)
    plt.subplot(6, 3, 8)
    planr_plot(edmd)
    plt.subplot(6, 3, 11)
    planr_plot(edmd_he)
    plt.subplot(6, 3, 14)
    planr_plot(node)
    plt.subplot(6, 3, 17)
    planr_plot(hnko)

    plt.subplot(6,3,3)
    energy_plot(true,hnn,t)
    plt.subplot(6,3,6)
    energy_plot(true, hnn_nu, t)
    plt.subplot(6, 3, 9)
    energy_plot(true, edmd, t)
    plt.subplot(6, 3, 12)
    energy_plot(true, edmd_he, t)
    plt.subplot(6, 3, 15)
    energy_plot(true, node, t)
    plt.subplot(6, 3, 18)
    energy_plot(true, hnko, t)
    plt.xlabel('Time', fontsize=fontsize)
    plt.legend(frameon=False,ncol=3,bbox_to_anchor=[0,-0.3],fontsize=10)

    plt.show()


# plot()