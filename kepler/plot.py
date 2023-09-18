import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl


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

def plot_8():  # X:true, Y:pred
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

    X = np.load('./data/true_300_0.03.npy')[:150]
    # Y = np.load('./data/noko_0.03.npy')
    Y = np.load('./data/hnko_300_0.03.npy')[:150]
    H = np.load('./data/hnn_300_0.03.npy')[:150]
    Z = np.load('./data/sympnet_300_0.03.npy')[:150]
    W = np.load('./data/edmd_300_0.03.npy')[:150]
    Q = np.load('./data/nc_300_0.03.npy')[:150]
    N = np.load('./data/hnko_noise_data.npy')[:,:150,:]
    print(X.shape,Y.shape,Z.shape,W.shape)
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.85, hspace=0.4, wspace=0.18)
    plt.subplot(131)
    plt.plot(H[:50, 0], H[:50, 1], color=colors[4], label='HNN', lw=global_lw)
    plt.plot(Z[100:,0],Z[100:,1],color=colors[0],label='SympNets', lw=global_lw)
    plt.plot(W[100:,0],W[100:,1],color=colors[1],label='EDMD', lw=global_lw)
    plt.plot(Q[100:, 0], Q[100:, 1], color=colors[5], label='DLKO', lw=global_lw)
    plt.plot(Y[100:,0],Y[100:,1], color=colors[2], label='HNKO', lw=global_lw)
    plt.plot(X[:50,0],X[:50,1], color='black', ls=(0, (4, 4)), label='Original dynamics', lw=global_lw)
    plt.xlabel(r'$q^1$',fontsize=fontsize,labelpad=labelpad)
    plt.ylabel(r'$q^2$',fontsize=fontsize, labelpad=labelpad-15)
    plt.yticks([-1,1.2],fontsize=labelsize)
    plt.xticks([-1,1.2],fontsize=labelsize)
    # plt.xlim(-1.2,1.3)
    # plt.ylim(-1.2, 1.3)
    plt.legend(ncol=6, bbox_to_anchor=[3.45, 1.2], fontsize=11, frameon=False)


    t = np.linspace(0,15,150)
    # plt.subplot(131)
    # plt.plot(t, coord_MSE(X,Z),c=colors[0],label='HNN', lw=global_lw)
    # plt.plot(t, coord_MSE(X,W),c=colors[1],label='EDMD', lw=global_lw)
    # plt.plot(t, coord_MSE(X,Y),c=colors[2],label='HNKO', lw=global_lw)
    # plt.plot(t, coord_MSE(X,X), c='black', ls=(0,(5,5)),label='Original dynamics', lw=global_lw)
    # plt.ylim(-0.01,0.11)
    # plt.xticks([0,15],fontsize = labelsize)
    # plt.yticks([0,0.1],fontsize = labelsize)
    # plt.ylabel('Prediction error',fontsize=fontsize, labelpad=labelpad)
    # plt.xlabel('Time',fontsize=fontsize,labelpad=labelpad)


    plt.subplot(332)
    plt.plot(t, k_energy(H), c=colors[4], label='HNN', lw=global_lw)
    plt.plot(t,k_energy(Z), c=colors[0], label='SympNets', lw=global_lw)
    plt.plot(t, k_energy(W), c=colors[1], label='EDMD', lw=global_lw)
    plt.plot(t, k_energy(Q), c=colors[5], label='DLKO', lw=global_lw)
    plt.plot(t, k_energy(Y), c=colors[2], label='HNKO', lw=global_lw)
    plt.plot(t, k_energy(X), c='black', ls=(0,(5,5)), label='Original dynamics', lw=global_lw)
    # plt.legend(ncol=4, bbox_to_anchor=[0.693, 1.2], fontsize=12, frameon=False)
    plt.ylim(0.25,1)
    plt.xticks([0, 15], fontsize=labelsize)
    plt.yticks([0.25,1.0],fontsize = labelsize)
    plt.ylabel(r'$E_k$',fontsize=fontsize, labelpad=labelpad-15)
    plt.subplot(335)
    plt.plot(t, p_energy(H), c=colors[4], lw=global_lw)
    plt.plot(t, p_energy(Z), c=colors[0], lw=global_lw)
    plt.plot(t, p_energy(W), c=colors[1],  lw=global_lw)
    plt.plot(t, p_energy(Q), c=colors[5],  lw=global_lw)
    plt.plot(t, p_energy(Y), c=colors[2], lw=global_lw)
    plt.plot(t, p_energy(X), c='black', ls=(0, (5, 5)),  lw=global_lw)
    plt.ylim(-1.6,-0.9)
    plt.xticks([0, 15], fontsize=labelsize)
    plt.yticks([-1.5,-1.0],fontsize = labelsize)
    plt.ylabel(r'$E_p$',fontsize=fontsize, labelpad=labelpad-18)
    plt.subplot(338)
    plt.plot(t, t_energy(H), c=colors[4], lw=global_lw)
    plt.plot(t, t_energy(Z), c=colors[0], lw=global_lw)
    plt.plot(t, t_energy(W), c=colors[1],  lw=global_lw)
    plt.plot(t, t_energy(Q), c=colors[5], lw=global_lw)
    plt.plot(t, t_energy(Y), c=colors[2], lw=global_lw)
    plt.plot(t, t_energy(X), c='black', ls=(0, (5, 5)), lw=global_lw)
    plt.xlabel('Time',fontsize=fontsize,labelpad=labelpad)
    plt.ylabel(r'$E$',fontsize=fontsize, labelpad=labelpad-10)
    plt.ylim(-1.1,0.1)
    plt.yticks([-1.0,0.0],fontsize = labelsize)
    plt.xticks([0,15],fontsize = labelsize)
    # plt.legend(frameon=False,loc=5)
    # plt.legend(ncol=4, bbox_to_anchor=[0.9, 1.2], fontsize=12, frameon=False)

    plt.subplot(133)
    c1,c2 = 'tab:blue','tab:red'
    lste,t_e,p_e,k_e = [],[],[],[]
    for i in range(11):
        lste.append(np.sqrt(np.mean(np.sum((X-N[i])[:,:2]**2,axis=1)))),t_e.append(np.mean(np.abs(t_energy(X)-t_energy(N[i]))))
    plt.plot(np.linspace(0,0.1,11),np.array(lste),color=c1,label='State',marker='$\circ$',markersize=8)
    # plt.scatter(np.linspace(0,0.1,11),np.array(lste),marker='o',color='',edgecolors=c1)
    plt.plot(np.linspace(0, 0.1, 11), np.array(t_e),color=c2,label='Energy',marker='$\Delta$',markersize=8)
    # plt.scatter(np.linspace(0, 0.1, 11), np.array(t_e), marker='o', color='', edgecolors=c2,label='Energy')
    plt.axvline(0.04,ls='--',color=colors[3],alpha=0.6)
    # plt.xlabel(r'$\sigma^2$',fontsize=fontsize,labelpad=labelpad+2)
    plt.text(0.06,-0.3,r'$\sigma^2$', fontsize=fontsize)
    plt.ylabel('Error',fontsize=fontsize, labelpad=labelpad)
    plt.yticks([0,2],fontsize=labelsize)
    plt.xticks([0,0.04,0.1],['0','0.04','0.1'],fontsize=labelsize)
    plt.legend()


    # plt.plot(X[:,0],X[:,1],label='True',color='black')
    # plt.plot(Y[:,0],Y[:,1],label='HNKO',color=colors[2])
    # plt.ylabel('Trajectory',fontsize=fontsize)
    # # plt.ylabel('y',fontsize=fontsize)
    # plt.title('HNKO',fontsize=fontsize)
    # plt.xlabel(r'$x$',fontsize=fontsize)
    # plt.xticks([-1,0,1],fontsize = labelsize)
    # plt.yticks([-1,0,1],fontsize = labelsize)
    # plt.legend(frameon=False)
    #
    # plt.subplot(242)
    # plt.plot(X[:,0],X[:,1],label='True',color='black')
    # plt.plot(Z[:,0],Z[:,1],label='HNN',color=colors[1])
    # plt.title('HNN',fontsize=fontsize)
    # plt.xticks([-1,0,1],fontsize = labelsize)
    # plt.yticks([-1,0,1],fontsize = labelsize)
    # plt.legend(frameon=False)
    # plt.subplot(243)
    # plt.plot(X[:,0],X[:,1],label='True',color='black')
    # plt.plot(W[:,0],W[:,1],label='EDMD',color=colors[0])
    # plt.title('EDMD',fontsize=fontsize)
    # plt.xticks([-1,0,1])
    # plt.yticks([-1,0,1])
    # plt.legend(frameon=False)
    # plt.subplot(244)
    # plt.plot(np.arange(len(coord_MSE(X,Y))),coord_MSE(X,Y),color=colors[2],label='HNKO')
    # plt.plot(np.arange(len(coord_MSE(X,Z))),coord_MSE(X,Z),color=colors[1],label='HNN')
    # plt.plot(np.arange(len(coord_MSE(X,W))),coord_MSE(X,W),color=colors[0],label='EDMD')
    # plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    # plt.yticks([0,0.05,0.1],fontsize = labelsize)
    # plt.title('MSE(Coordinates)',fontsize=fontsize)
    # plt.legend(frameon=False)
    # plt.subplot(245)
    # plt.plot(np.arange(len(k_energy(Y))),k_energy(Y),label='Kinetic',color=colors[2],ls='--')
    # plt.plot(np.arange(len(p_energy(Y))),p_energy(Y),label='Total',color=colors[2],ls='dotted')
    # plt.plot(np.arange(len(t_energy(Y))),t_energy(Y),label='Potential',color=colors[2],ls='-')
    # plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    # plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    # plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    # plt.ylabel('Energy',fontsize=fontsize)
    # plt.xlabel('Time',fontsize=fontsize)
    # plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    # plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    # plt.legend(frameon=False,loc=5)
    # plt.subplot(246)
    # plt.plot(np.arange(len(k_energy(Z))),k_energy(Z),label='Kinetic',color=colors[1],ls='--')
    # plt.plot(np.arange(len(p_energy(Z))),p_energy(Z),label='Total',color=colors[1],ls='dotted')
    # plt.plot(np.arange(len(t_energy(Z))),t_energy(Z),label='Potential',color=colors[1],ls='-')
    # plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    # plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    # plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    # plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    # plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    # plt.legend(frameon=False,loc=5)
    # plt.subplot(247)
    # plt.plot(np.arange(len(k_energy(W))),k_energy(W),label='Kinetic',color=colors[0],ls='--')
    # plt.plot(np.arange(len(p_energy(W))),p_energy(W),label='Total',color=colors[0],ls='dotted')
    # plt.plot(np.arange(len(t_energy(W))),t_energy(W),label='Potential',color=colors[0],ls='-')
    # plt.plot(np.arange(len(k_energy(X))),k_energy(X),color='black',ls='--')
    # plt.plot(np.arange(len(p_energy(X))),p_energy(X),color='black',ls='dotted')
    # plt.plot(np.arange(len(t_energy(X))),t_energy(X),color='black',ls='-')
    # plt.yticks([-1.5,0.,1.5],fontsize = labelsize)
    # plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    # plt.legend(frameon=False,loc=5)
    # plt.subplot(248)
    # plt.plot(np.arange(len(t_energy(X))),np.sqrt((t_energy(X)-t_energy(Y))**2),color=colors[2],label='HNKO')
    # plt.plot(np.arange(len(t_energy(Y))),np.sqrt((t_energy(X)-t_energy(Z))**2),color=colors[1],label='HNN')
    # plt.plot(np.arange(len(t_energy(Z))),np.sqrt((t_energy(X)-t_energy(W))**2),color=colors[0],label='EDMD')
    # plt.xticks([0,25,50],[0,2.5,5],fontsize = labelsize)
    # plt.yticks([0,0.1,0.2],fontsize = labelsize)
    # plt.title('MSE(Total Energy)',fontsize=fontsize)
    # plt.legend(frameon=False)
    plt.show()
plot_8()

