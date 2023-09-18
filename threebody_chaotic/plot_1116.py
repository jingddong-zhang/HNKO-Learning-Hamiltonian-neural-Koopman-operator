import numpy as np
import matplotlib.pyplot as plt

global_lw = 1.0
legendsize = 7
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
def hami(x):
    x1,y1,x2,y2,x3,y3,a1,b1,a2,b2,a3,b3 = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],x[:,8],x[:,9],x[:,10],x[:,11]
    h = (a1**2+b1**2+a2**2+b2**2+a3**2+b3**2)/2-1/(np.sqrt((x1-x2)**2+(y1-y2)**2))\
            -1/(np.sqrt((x1-x3)**2+(y1-y3)**2))-1/(np.sqrt((x3-x2)**2+(y3-y2)**2))
    return h

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

def subplot(data,leg=False):
    co1 = colors[0]
    co2 = colors[1]
    co3 = colors[2]
    width = global_lw
    s = 25
    style = 'solid'
    plt.plot(data[:,0],data[:,1],color=co1,ls=style,lw=width, label=r'$\bm{q}_1$')
    plt.plot(data[:,2],data[:,3],color=co2,ls=style,lw=width, label=r'$\bm{q}_2$')
    plt.plot(data[:,4],data[:,5],color=co3,ls=style,lw=width, label=r'$\bm{q}_3$')

    plt.scatter(data[-1,0],data[-1,1],color=co1,marker='o',s=s)
    plt.scatter(data[-1,2], data[-1,3], color=co2, marker='o', s=s)
    plt.scatter(data[-1, 4], data[-1, 5], color=co3, marker='o', s=s)

    # labelpad_x = 9
    # labelpad_y = 5
    # fontsize = 12
    # plt.xlabel(r'$\bm{q}_i^1$', fontsize=fontsize, labelpad=labelpad_x)
    # plt.ylabel(r'$\bm{q}_i^2$', fontsize=fontsize, labelpad=labelpad_y)
    if leg:
        # plt.legend(ncol=3, bbox_to_anchor=[2.46, -1.55], fontsize=9, frameon=False)
        plt.legend(ncol=3, bbox_to_anchor=[2.2, 1.2], fontsize=12, frameon=False)
        # plt.legend(ncol=1, fontsize=8.5,framealpha=0.5,loc=4)



def set_title(title):
    plt.title(title)

def plot():
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Nsimsun,Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    # matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 15
    labelsize= 12

    '''
    data size: (1800,12) 
    1800: time interval on [0,90], 1800 uniform samples
    12:   state q_x1,q_y1,q_x2,q_y2,q_x3,q_y3,p_x1,p_y1,p_x2,p_y2,p_x3,p_y3
    here we choose time intrval on [0,50] for plot, that is, data[:1000,:]
    '''

    true = np.load('./data/true_chaos_7_200.npy')
    hnn = np.load('./data/hnn_chaos_7_200_0.01.npy')
    hnko = np.load('./data/hnko_chaos_7_200_0.01.npy')
    edmd = np.load('./data/edmd_chaos_7_200_0.01.npy')
    # node = np.load('./data/node_30_600.npy')

    '''
    plot
    '''
    # fig = plt.figure(figsize=(4.8, 3.2))
    # plt.subplots_adjust(left=0.01, bottom=0.14, right=0.98, top=0.98, hspace=0.4, wspace=0.4)
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.94, top=0.88, hspace=0.27, wspace=0.3)

    labelpad_x = -7
    labelpad_y = -14
    plt.subplot(141)
    subplot(true[:200],leg=False)
    # plt.title('Truth',fontsize=fontsize)
    plt.xticks([-1,1],fontsize=labelsize)
    plt.yticks([-1,1],fontsize=labelsize)
    plt.ylabel(r'$q_i^2$', fontsize=fontsize, labelpad=labelpad_y)

    plt.subplot(142)
    subplot(hnko[:200],leg=True)
    # plt.xlabel(r'$q_i^1$', fontsize=fontsize, labelpad=labelpad_x)
    plt.text(1.2,-1.5,r'$q_i^1$', fontsize=fontsize)
    plt.xticks([-1,1],fontsize=labelsize)
    plt.yticks([-1,1],fontsize=labelsize)
    # plt.title('HNKO',fontsize=fontsize)
    # plt.xlabel('HNKO', fontsize=fontsize, labelpad=labelpad_x)

    plt.subplot(144)
    subplot(edmd[:200])
    plt.xticks([-3e17, 5e17],['-3','5e17'],fontsize=labelsize)
    plt.yticks([-3e17, 3e17],['-3','3e17'],fontsize=labelsize)
    # plt.title('EDMD',fontsize=fontsize)
    # plt.xlabel('EDMD', fontsize=fontsize, labelpad=labelpad_x)

    plt.subplot(143)
    subplot(hnn[:200])
    # plt.title('HNN',fontsize=fontsize)
    plt.xticks([-3, 3],fontsize=labelsize)
    plt.yticks([-3, 3],fontsize=labelsize)
    # plt.xlabel(r'$q_i^1$', fontsize=fontsize, labelpad=labelpad_x)
    # plt.ylabel(r'$q_i^2$', fontsize=fontsize, labelpad=labelpad_y)
    # labelpad = -3

    # ax_1 = plt.subplot(235)
    # ax_1.plot(t,state_error(true,hnko),label='HNKO', lw=global_lw)
    # ax_1.plot(t,state_error(true,edmd),label='EDMD', lw=global_lw)
    # ax_1.plot(t,state_error(true,hnn),label='HNN', lw=global_lw)
    # plt.xticks([0,25,50],fontsize=labelsize)
    # plt.ylim(-0.1,1.1)
    # plt.yticks([0,1],fontsize=labelsize)
    # # plt.yticks([0, 8, 16], fontsize=labelsize)
    # # ax_1.set_xlabel('Time',fontsize=fontsize)
    # ax_1.set_ylabel('Prediction error',fontsize=fontsize-1, labelpad=labelpad)
    # plt.legend(fontsize=legendsize, loc=1,framealpha=0.5)
    # plt.xlabel('Time',fontsize=fontsize)

    # ax_2 = plt.subplot(236)
    # plt.plot(t,hami(hnko),label='HNKO', lw=global_lw)
    # plt.plot(t,hami(edmd),label='EDMD', lw=global_lw)
    # plt.plot(t,hami(hnn),label='HNN', lw=global_lw)
    # plt.plot(t,hami(true),color='black',label='Original dynamics',ls=(0,(3,3)), lw=global_lw)
    # plt.ylim([-3,5])
    # plt.yticks([])
    # # plt.legend(ncol=4,bbox_to_anchor=[0,-0.3],fontsize=10,frameon=False)
    # # plt.legend(ncol=4, bbox_to_anchor=[0, 0], fontsize=10, frameon=False)
    # plt.legend(fontsize=legendsize, loc=2,framealpha=0.5)
    # # plt.xlabel('Time',fontsize=fontsize)
    # plt.text(-15, -5.0, 'Time', fontsize=fontsize)
    #
    # ax_3 = ax_2.twinx()
    # plt.xticks([0,25,50],fontsize=labelsize)
    # plt.yticks([-2,0,5],fontsize=labelsize)
    # plt.ylabel('Energy',fontsize=fontsize-1, labelpad=labelpad+9,rotation=-90)

    plt.show()

# subplot in the Figure(e)

plot()