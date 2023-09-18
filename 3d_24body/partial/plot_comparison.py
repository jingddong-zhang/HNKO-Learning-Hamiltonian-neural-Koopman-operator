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


markersize = 15
def subplot1(data, co2,co1=[0.9,0.9,0.9],fontsize=18,labelpad=-10,xlabel=False,ylabel=False):
    data = data[:140]

    for i in range(1,len(data)):
        coef = i/len(data)
        # co1,co2 = colors[0],colors[2]
        # co1 = [0.1,0.1,0.1]
        # co2 = [0.9,0.8,0.7]
        print(co1[0],co2[0])
        color = [coef*co1[0]+(1-coef)*co2[0],
                 coef*co1[1]+(1-coef)*co2[1],
                 coef*co1[2]+(1-coef)*co2[2]]
        plt.scatter(data[i:i+1, 0], data[i:i+1, 1], c=color)
    plt.plot(data[0:1, 0], data[0:1, 1], c=co2,marker='*',markersize=markersize)
    # plt.scatter(data[:,0],data[:,1],c=co2)
    # plt.title(title,fontsize = fontsize)
    if xlabel:
        plt.xlabel(r'$x$', fontsize=fontsize, labelpad=labelpad)
    if ylabel:
        plt.ylabel(r'$y$', fontsize=fontsize, labelpad=labelpad)
    # plt.xticks([-0.2,0.8])
    # plt.yticks([-0.2,0.8])
    plt.xticks([])
    plt.yticks([])

def subplot2(data, co2,co1=[0.9,0.9,0.9]):
    data = data[:140]
    for i in range(1,len(data)):
        coef = i/len(data)
        # co1,co2 = colors[0],colors[2]
        # co1 = [0.1,0.1,0.1]
        # co2 = [0.9,0.8,0.7]
        print(co1[0],co2[0])
        color = [coef*co1[0]+(1-coef)*co2[0],
                 coef*co1[1]+(1-coef)*co2[1],
                 coef*co1[2]+(1-coef)*co2[2]]
        plt.scatter(data[i:i+1, 32], data[i:i+1, 33], c=color)
    plt.plot(data[0:1, 32], data[0:1, 33], c=co2,marker='*',markersize=markersize)
    # plt.scatter(data[:,16],data[:,17],c=co)
    plt.xticks([])
    plt.yticks([])


def subplot3(data, co2,co1=[0.9,0.9,0.9]):
    data = data[:140]
    for i in range(1,len(data)):
        coef = i/len(data)
        # co1,co2 = colors[0],colors[2]
        # co1 = [0.1,0.1,0.1]
        # co2 = [0.9,0.8,0.7]
        print(co1[0],co2[0])
        color = [coef*co1[0]+(1-coef)*co2[0],
                 coef*co1[1]+(1-coef)*co2[1],
                 coef*co1[2]+(1-coef)*co2[2]]
        plt.scatter(data[i:i+1, -4], data[i:i+1, -3], c=color)
    plt.plot(data[0:1, -4], data[0:1, -3], c=co2,marker='*',markersize=markersize)
    # plt.scatter(data[:, -4], data[:, -3], c=co)
    plt.xticks([])
    plt.yticks([])



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
    fontsize = 18
    labelsize= 15


    true = np.load('./data/true_15_5_1000_symplectic_r7_a9.npy')



    '''
    plot
    '''
    # fig = plt.figure(figsize=(4.8, 3.2))
    # plt.subplots_adjust(left=0.01, bottom=0.14, right=0.98, top=0.98, hspace=0.4, wspace=0.4)
    # fig = plt.figure(figsize=(5.2, 2.8))
    # plt.subplots_adjust(left=0.07, bottom=0.11, right=0.94, top=0.98, hspace=0.27, wspace=0.27)

    fig, axs = plt.subplots(3, 6)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.95, hspace=0.15, wspace=0.2)
    labelpad_x = -7
    labelpad_y = -14


    plt.subplot(361)
    subplot1(true,[0,0,0])
    plt.title('Truth',fontsize=labelsize,c='black')
    plt.ylabel(r'body-$1$',fontsize=labelsize)
    # plt.xlabel(r'$x$', fontsize=labelsize)

    hnn = np.load('./data/hnn_5_1000_tanh_0.01_4_64.npy')
    symla = np.load('./data/sympnet_15body_5_1000_0.01_LA.npy')
    symg = np.load('./data/sympnet_15body_5_1000_0.01_G_128.npy')
    dlko = np.load('./data/dlko_160_0.01.npy')
    hnko = np.load('./data/hnko_70_40_5_1000_0.01.npy')




    plt.subplot(362)
    subplot1(hnko,colors[2])
    plt.title('HNKO',fontsize=labelsize,c=colors[2])

    plt.subplot(363)
    subplot1(dlko,colors[5])
    plt.title('DLKO', fontsize=labelsize,c=colors[5])
    plt.subplot(364)
    subplot1(hnn,colors[4])
    plt.title('HNN', fontsize=labelsize,c=colors[4])

    plt.subplot(365)
    subplot1(symg,colors[7])
    plt.title('G-SympNets', fontsize=labelsize,c=colors[7])

    plt.subplot(366)
    subplot1(symla,colors[0])
    plt.title('LA-SympNets', fontsize=labelsize,c=colors[0])





    plt.subplot(367)
    subplot2(true,[0,0,0])

    plt.ylabel(r'body-$9$',fontsize=labelsize)

    plt.subplot(368)
    subplot2(hnko,colors[2])

    plt.subplot(3,6,9)
    subplot2(dlko,colors[5])

    plt.subplot(3,6,10)
    subplot2(hnn,colors[4])

    plt.subplot(3,6,11)
    subplot2(symg,colors[7])

    plt.subplot(3,6,12)
    subplot2(symla,colors[0])

    plt.subplot(3,6,13)
    subplot3(true,[0,0,0])

    plt.ylabel(r'body-$15$',fontsize=labelsize)

    plt.subplot(3,6,14)
    subplot3(hnko,colors[2])

    plt.subplot(3,6,15)
    subplot3(dlko,colors[5])

    plt.subplot(3,6,16)
    subplot3(hnn,colors[4])

    plt.subplot(3,6,17)
    subplot3(symg,colors[7])

    plt.subplot(3,6,18)
    subplot3(symla,colors[0])

    plt.show()


plot()