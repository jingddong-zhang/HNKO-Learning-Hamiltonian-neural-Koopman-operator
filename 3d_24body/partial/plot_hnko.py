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
    labelsize = 15

    true = np.load('./data/true_15_5_1000_symplectic_r7_a9.npy')
    hnko0 = np.load('./data/hnko_70_40_5_1000_0.0.npy')
    hnko1 = np.load('./data/hnko_70_40_5_1000_0.01.npy')
    hnko2 = np.load('./data/hnko_70_40_5_1000_0.02.npy')
    hnko3 = np.load('./data/hnko_70_40_5_1000_0.03.npy')

    fig, axs = plt.subplots(3, 5)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.95, hspace=0.15, wspace=0.2)

    plt.subplot(351)
    subplot1(true, [0, 0, 0])
    plt.title('True',fontsize=fontsize)
    plt.ylabel(r'body-$1$',fontsize=fontsize)

    plt.subplot(352)
    subplot1(hnko0,colors[2])
    plt.title(r'$\sigma=0$', fontsize=fontsize)

    plt.subplot(353)
    subplot1(hnko1,colors[3])
    plt.title(r'$\sigma=0.01$', fontsize=fontsize)
    # plt.xlabel(r'$x$',fontsize=fontsize)

    plt.subplot(354)
    subplot1(hnko2,colors[4])
    plt.title(r'$\sigma=0.02$', fontsize=fontsize)

    plt.subplot(355)
    subplot1(hnko3,colors[5])
    plt.title(r'$\sigma=0.03$', fontsize=fontsize)

    plt.subplot(356)
    subplot2(true, [0, 0, 0])
    plt.ylabel(r'body-$9$', fontsize=fontsize)

    plt.subplot(357)
    subplot2(hnko0, colors[0])

    plt.subplot(358)
    subplot2(hnko1, colors[3])

    plt.subplot(359)
    subplot2(hnko2, colors[4])

    plt.subplot(3,5,10)
    subplot2(hnko3, colors[5])

    plt.subplot(3,5,11)
    subplot3(true, [0, 0, 0])
    plt.ylabel(r'body-$15$', fontsize=fontsize)

    plt.subplot(3,5,12)
    subplot3(hnko0, colors[2])

    plt.subplot(3,5,13)
    subplot3(hnko1, colors[3])

    plt.subplot(3,5,14)
    subplot3(hnko2, colors[4])

    plt.subplot(3,5,15)
    subplot3(hnko3, colors[5])




plot()
plt.show()