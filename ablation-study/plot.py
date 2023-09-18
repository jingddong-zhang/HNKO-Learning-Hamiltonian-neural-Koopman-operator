import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import torch

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

def t_energy(X):  #total energy
    x,y,a,b = X[:,0],X[:,1],X[:,2],X[:,3]
    h = (a**2+b**2)/2-1/(np.sqrt(x**2+y**2))
    return h

def coord_MSE(X,Y): # X:true, Y:pred, [:,2]
    err = np.sum((X-Y)[:,:2]**2,axis=1)
    return err
def k_energy(X):  #kinetic energy
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = (a**2+b**2)/2
    return h
def p_energy(X):  #potential energy
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = -1/(np.sqrt(x**2+y**2))
    return h
def angular(X):  #Angular Momentum
    x,y,a,b=X[:,0],X[:,1],X[:,2],X[:,3]
    h = x*b-y*a
    return h

global_lw = 2.0
legendsize = 7

def plot():  # X:true, Y:pred
    import matplotlib
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    # matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 15
    labelsize= 12
    labelpad = -9

    N = 200
    X = np.load('./data/true_300_0.03.npy')[:N]
    Y1 = np.load('./data/hnko_withV_300_0.03.npy')[:N]
    Y2 = np.load('./data/hnko_withV_300_0.05.npy')[:N]
    Z1 = np.load('./data/hnko_withoutV_300_0.03.npy')[:N]
    Z2 = np.load('./data/hnko_withoutV_300_0.05.npy')[:N]


    # fig = plt.figure(figsize=(8, 3))
    # plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.85, hspace=0.4, wspace=0.18)


    fig, axs = plt.subplots(1, 5)
    fig.set_figheight(3)
    fig.set_figwidth(10)
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.91, hspace=0.35, wspace=0.35)

    plt.subplot(151)
    plt.plot(np.arange(len(k_energy(X))),k_energy(X),label='kinetic')
    plt.plot(np.arange(len(p_energy(X))),p_energy(X),label='total')
    plt.plot(np.arange(len(t_energy(X))),t_energy(X),label='potential')
    plt.legend(fontsize=labelsize,frameon=False,bbox_to_anchor=(0.11, 0.4))
    plt.yticks([-1.5,0.5],fontsize=labelsize)
    plt.xticks([0,200],fontsize=labelsize)
    plt.title('Truth')

    plt.subplot(152)
    plt.plot(np.arange(len(k_energy(Y1))),k_energy(Y1),label='kinetic')
    plt.plot(np.arange(len(p_energy(Y1))),p_energy(Y1),label='total')
    plt.plot(np.arange(len(t_energy(Y1))),t_energy(Y1),label='potential')
    plt.yticks([-1.5,0.5],fontsize=labelsize)
    plt.xticks([0,200],fontsize=labelsize)
    plt.title(r'With V, $\sigma^2=0.03$')



    plt.subplot(153)
    plt.plot(np.arange(len(k_energy(Z1))),k_energy(Z1),label='kinetic')
    plt.plot(np.arange(len(p_energy(Z1))),p_energy(Z1),label='total')
    plt.plot(np.arange(len(t_energy(Z1))),t_energy(Z1),label='potential')
    plt.title(r'Without V, $\sigma^2=0.03$')
    plt.yticks([-1.5,0.5],fontsize=labelsize)
    plt.xticks([0,200],fontsize=labelsize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-10)

    plt.subplot(154)
    plt.plot(np.arange(len(k_energy(Y2))),k_energy(Y2),label='kinetic')
    plt.plot(np.arange(len(p_energy(Y2))),p_energy(Y2),label='total')
    plt.plot(np.arange(len(t_energy(Y2))),t_energy(Y2),label='potential')
    plt.yticks([-1.5,0.5],fontsize=labelsize)
    plt.xticks([0,200],fontsize=labelsize)
    plt.title(r'With V, $\sigma^2=0.05$')

    plt.subplot(155)
    plt.plot(np.arange(len(k_energy(Z2))),k_energy(Z2),label='kinetic')
    plt.plot(np.arange(len(p_energy(Z2))),p_energy(Z2),label='total')
    plt.plot(np.arange(len(t_energy(Z2))),t_energy(Z2),label='potential')
    plt.yticks([-1.5,0.5],fontsize=labelsize)
    plt.xticks([0,200],fontsize=labelsize)
    plt.title(r'Without V, $\sigma^2=0.05$')

    # plt.plot(X[100:,0],X[100:,1],color=colors[0],label='True', lw=global_lw)
    # plt.plot(Y2[100:,0],Y2[100:,1],color=colors[1],label='With V', lw=global_lw)
    # plt.plot(Z2[100:, 0], Z2[100:, 1], color=colors[5], label='Without V', lw=global_lw)
    # plt.plot(Y[100:,0],Y[100:,1], color=colors[2], label='HNKO', lw=global_lw)
    # plt.plot(X[:50,0],X[:50,1], color='black', ls=(0, (4, 4)), label='Original dynamics', lw=global_lw)
    # plt.xlabel(r'$q^1$',fontsize=fontsize,labelpad=labelpad)
    # plt.ylabel(r'$q^2$',fontsize=fontsize, labelpad=labelpad-15)
    # plt.yticks([-1,1.2],fontsize=labelsize)
    # plt.xticks([-1,1.2],fontsize=labelsize)
    # plt.legend(ncol=5, bbox_to_anchor=[2.95, 1.2], fontsize=11, frameon=False)
    #
    #
    # t = np.linspace(0,15,150)
    #
    #
    # plt.subplot(332)
    # plt.plot(t,k_energy(Z), c=colors[0], label='SympNet', lw=global_lw)
    # plt.plot(t, k_energy(W), c=colors[1], label='EDMD', lw=global_lw)
    # plt.plot(t, k_energy(Q), c=colors[5], label='DLKO', lw=global_lw)
    # plt.plot(t, k_energy(Y), c=colors[2], label='HNKO', lw=global_lw)
    # plt.plot(t, k_energy(X), c='black', ls=(0,(5,5)), label='Original dynamics', lw=global_lw)
    # # plt.legend(ncol=4, bbox_to_anchor=[0.693, 1.2], fontsize=12, frameon=False)
    # plt.ylim(0.25,1)
    # plt.xticks([0, 15], fontsize=labelsize)
    # plt.yticks([0.25,1.0],fontsize = labelsize)
    # plt.ylabel(r'$E_k$',fontsize=fontsize, labelpad=labelpad-15)
    # plt.subplot(335)
    # plt.plot(t, p_energy(Z), c=colors[0], lw=global_lw)
    # plt.plot(t, p_energy(W), c=colors[1],  lw=global_lw)
    # plt.plot(t, p_energy(Q), c=colors[5],  lw=global_lw)
    # plt.plot(t, p_energy(Y), c=colors[2], lw=global_lw)
    # plt.plot(t, p_energy(X), c='black', ls=(0, (5, 5)),  lw=global_lw)
    # plt.ylim(-1.6,-0.9)
    # plt.xticks([0, 15], fontsize=labelsize)
    # plt.yticks([-1.5,-1.0],fontsize = labelsize)
    # plt.ylabel(r'$E_p$',fontsize=fontsize, labelpad=labelpad-18)
    # plt.subplot(338)
    # plt.plot(t, t_energy(Z), c=colors[0], lw=global_lw)
    # plt.plot(t, t_energy(W), c=colors[1],  lw=global_lw)
    # plt.plot(t, t_energy(Q), c=colors[5], lw=global_lw)
    # plt.plot(t, t_energy(Y), c=colors[2], lw=global_lw)
    # plt.plot(t, t_energy(X), c='black', ls=(0, (5, 5)), lw=global_lw)
    # plt.xlabel('Time',fontsize=fontsize,labelpad=labelpad)
    # plt.ylabel(r'$E$',fontsize=fontsize, labelpad=labelpad-10)
    # plt.ylim(-1.1,0.1)
    # plt.yticks([-1.0,0.0],fontsize = labelsize)
    # plt.xticks([0,15],fontsize = labelsize)
    # # plt.legend(frameon=False,loc=5)
    # # plt.legend(ncol=4, bbox_to_anchor=[0.9, 1.2], fontsize=12, frameon=False)
    #
    # plt.subplot(133)
    # c1,c2 = 'tab:blue','tab:red'
    # lste,t_e,p_e,k_e = [],[],[],[]
    # for i in range(11):
    #     lste.append(np.sqrt(np.mean(np.sum((X-N[i])[:,:2]**2,axis=1)))),t_e.append(np.mean(np.abs(t_energy(X)-t_energy(N[i]))))
    # plt.plot(np.linspace(0,0.1,11),np.array(lste),color=c1,label='State',marker='$\circ$',markersize=8)
    # # plt.plot(np.linspace(0,0.1,11),np.array(lste),marker='o',color='',edgecolors=c1)
    # plt.plot(np.linspace(0, 0.1, 11), np.array(t_e),color=c2,label='Energy',marker='$\Delta$',markersize=8)
    # # plt.plot(np.linspace(0, 0.1, 11), np.array(t_e), marker='o', color='', edgecolors=c2,label='Energy')
    # plt.axvline(0.04,ls='--',color=colors[3],alpha=0.6)
    # # plt.xlabel(r'$\sigma^2$',fontsize=fontsize,labelpad=labelpad+2)
    # plt.text(0.06,-0.3,r'$\sigma^2$', fontsize=fontsize)
    # plt.ylabel('Error',fontsize=fontsize, labelpad=labelpad)
    # plt.yticks([0,2],fontsize=labelsize)
    # plt.xticks([0,0.04,0.1],['0','0.04','0.1'],fontsize=labelsize)
    # plt.legend()


    plt.show()
plot()

