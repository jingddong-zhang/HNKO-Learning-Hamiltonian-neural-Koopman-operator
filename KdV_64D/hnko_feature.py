import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from functions import *

class K_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(K_Net, self).__init__()
        # torch.manual_seed(2)
        self.recurrent_kernel = nn.Linear(n_input, n_output, bias=False)
        geotorch.orthogonal(self.recurrent_kernel, "weight")
        self.reset_parameters()

    def reset_parameters(self):
        # The manifold class is under `layer.parametrizations.tensor_name[0]`
        M = self.recurrent_kernel.parametrizations.weight[0]
        # Every manifold has a convenience sample method, but you can use your own initializer
        self.recurrent_kernel.weight = M.sample("uniform")

    def forward(self, data):
        return self.recurrent_kernel(data)


class lift_Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(lift_Net, self).__init__()
        # torch.manual_seed(2)
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
            nn.Linear(n_hidden,n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.k = torch.tensor(2.5, requires_grad=True)
        self.v = torch.randn([4+lift_d,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out


class Decoder(nn.Module):

    def __init__(self,n_hidden):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4+lift_d, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            # nn.Linear(40, 40),
            # nn.Tanh(),
            nn.Linear(n_hidden,4)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

class kepler(nn.Module):
    dim = 4
    def forward(self, t, X):
        # x: [b_size, d_dim]
        dx = torch.zeros_like(X)
        x,y,a,b = X[:,0],X[:,1],X[:,2],X[:,3]
        dx[:, 0] = a
        dx[:, 1] = b
        dist = (x**2+y**2)**1.5
        dx[:, 2] = -x/dist
        dx[:, 3] = -y/dist
        return dx

n = 100
# y0 = torch.tensor([[0.0,1.0,1.,0.0]])
y0 = torch.tensor([[1.0,0.0,0.,0.9]])
t = torch.linspace(0, 5, n)
# y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
true_y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]


'''
For learning 
'''
sigma = 0.03
lift_d = 19 # 19
q = 15  # 13
D_in = 12+lift_d # input dimension
H1 = 10*D_in
D_out = 12+lift_d # output dimension

out_iters = 0
eigen_list=[]
def generate():
    start = timeit.default_timer()
    torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    g1 = lift_Net(4,32,4+lift_d)
    Dec = Decoder(32)
    i = 0
    max_iters = 10000
    learning_rate = 0.005
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in g1.parameters()]+[g1.k]+[g1.v]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)
    Loss = []
    while i < max_iters:
        # break
        lift_y = g1(y)
        dec_y = Dec(lift_y)
        X1,X2 = lift_y[:-1],lift_y[1:]

        v = g1.v/torch.sqrt(torch.sum(g1.v**2,dim=0))
        V = torch.mm(v.T,v)-torch.eye(q)
        # embed_y = torch.cat((y,g1(y)),dim=1)
        loss = torch.sum((X2-model(X1))**2) \
               +torch.sum((dec_y-y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)


        print(i, "loss=", loss.item())
        # print(out_iters,i, "loss=", loss.item(),g1.k,g1.v[:3])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        Loss.append(loss.item())
        if loss.item() == min(Loss):
            best_K = model.recurrent_kernel.weight.detach().numpy()
            # print(best_K.shape)
            best_enc = g1.state_dict()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    # K = model.recurrent_kernel.weight.detach().numpy()

    K = best_K
    np.save('./data/feature_data/koopman_{}_{}'.format(D_in,sigma),K)
    g1.load_state_dict(best_enc)
    enc = g1(y).detach().numpy()
    np.save('./data/feature_data/enc_{}_{}'.format(D_in,sigma),enc)
    # torch.save(best_enc,'./data/feature_data/enc_{}_{}.pkl'.format(D_in,sigma))
    # K = np.load('./data/koopman_0.0.npy')


    lift_y = g1(true_y)
    dec_y = Dec(lift_y)[:-1]
    y = y.detach().numpy()
    dec_y=dec_y.detach().numpy()

    n = 300
    Y = np.zeros([n,len(K)]) #NOKO
    Y[0,:]=lift_y.detach().numpy()[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()
    # np.save('./data/hnko_300_0.03',pred_y)
    Y = pred_y


    # plt.subplot(131)
    # plt.scatter(np.arange(len(k_energy(Y))),k_energy(Y),label='kinetic')
    # plt.scatter(np.arange(len(p_energy(Y))),p_energy(Y),label='total')
    # plt.scatter(np.arange(len(t_energy(Y))),t_energy(Y),label='potential')
    # plt.subplot(132)
    # plt.scatter(np.arange(len(k_energy(Y))),k_energy(Y),label='kinetic')
    # plt.scatter(np.arange(len(p_energy(Y))),p_energy(Y),label='total')
    # plt.scatter(np.arange(len(t_energy(Y))),t_energy(Y),label='potential')
    # plt.subplot(133)
    # K =K
    # eigenvalue, featurevector = np.linalg.eig(K)
    #
    # print("特征值：", np.real(eigenvalue))
    # print(featurevector.shape)
    # plt.bar(np.arange(len(K)), np.sort((featurevector[:, 2])))


    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369


def normalize(data):
    data = np.sort(np.abs(data))[::-1]
    return data/np.max(np.abs(data))
def check():
    dim = 17
    model = K_Net(dim, dim)
    K =  model.recurrent_kernel.weight.detach().numpy()
    eigenvalue, featurevector = np.linalg.eig(K)
    print("特征值：", np.real(eigenvalue))


def plot():
    fontsize = 15
    ticksize = 12
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    import matplotlib
    matplotlib.rcParams.update(rc_fonts)

    fig = plt.figure(figsize=(10,4))
    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.9, hspace=0.95, wspace=0.25)
    plt.subplot(131)
    sigma = 0.03
    K = np.load('./data/feature_data/koopman_31_0.03_threebody.npy')
    enc = np.load('./data/feature_data/enc_31_0.03_threebody.npy')
    print(K.shape,enc.shape)
    eigenvalue, featurevector = np.linalg.eig(K)
    print("特征值：", np.real(eigenvalue))
    print(featurevector.shape,enc.shape)
    print(featurevector[:,-1])
    weight = np.real(featurevector[:,-1])
    quantity = np.sum(weight*enc,axis=1)
    print(quantity.shape)
    var_list = []
    for i in range(len(K)):
        var_list.append(np.std(enc[:,i])**2)
    value_list = []
    value_list.append(np.std(quantity)**2)
    value_list += [var_list[i] for i in np.argsort(np.array(var_list))[::-1][:7]]
    value_list = [np.log(x) for x in value_list]
    index = ['$g_c$','$g_1$','$g_2$','$g_3$','$g_4$','$g_5$','$g_6$','$g_7$']
    color_list = ['k']+['gray' for _ in range(len(index)-1)]
    bar_width = 0.5
    plt.bar(np.arange(len(index))+bar_width*0,value_list,color=color_list, width=bar_width)#, color=seven_color[0], alpha=0.6, lw=2.0, edgecolor=seven_color[0],label='No Noise')
    plt.ylim(-15,0.5)
    plt.yticks([-15,0],fontsize=ticksize)
    plt.xticks(np.arange(len(index))+bar_width*0,index,fontsize=12)
    plt.xlabel('Component',fontsize=fontsize)
    plt.ylabel(r'$\log(\text{Temporal Variance})$',fontsize=fontsize,labelpad=-25)
    plt.title('Threebody',fontsize=fontsize)
    plt.scatter(np.arange(1),[-13],marker='*',color='red',s=50,label='Hamiltonian')
    plt.legend(fontsize=fontsize,loc=4,frameon=True)

    plt.subplot(132)
    K = np.load('./data/feature_data/koopman_7_0.0_kepler.npy')
    enc = np.load('./data/feature_data/enc_7_0.0_kepler.npy')
    print(K.shape, enc.shape)
    eigenvalue, featurevector = np.linalg.eig(K)
    print("特征值：", np.real(eigenvalue))
    print(featurevector.shape, enc.shape)
    print(featurevector[:, -1])
    weight = np.real(featurevector[:, -1])
    quantity = np.sum(weight * enc, axis=1)
    print(quantity.shape)
    var_list = []
    for i in range(len(K)):
        var_list.append(np.std(enc[:, i]) ** 2)
    value_list = []
    value_list.append(np.std(quantity) ** 2)
    value_list += [var_list[i] for i in np.argsort(np.array(var_list))[::-1][:7]]
    value_list = [np.log(x) for x in value_list]
    index = ['$g_c$', '$g_1$', '$g_2$', '$g_3$', '$g_4$', '$g_5$', '$g_6$', '$g_7$']
    color_list = ['k'] + ['gray' for _ in range(7)]
    bar_width = 0.5
    plt.bar(np.arange(len(index)) + bar_width * 0, value_list, color=color_list,
            width=bar_width)  # , color=seven_color[0], alpha=0.6, lw=2.0, edgecolor=seven_color[0],label='No Noise')
    plt.ylim(-8, 0.5)
    plt.yticks([-8, 0], fontsize=ticksize)
    plt.xticks(np.arange(len(index)) + bar_width * 0, index, fontsize=12)
    plt.xlabel('Component', fontsize=fontsize)
    plt.ylabel(r'$\log(\text{Temporal Variance})$', fontsize=fontsize, labelpad=-20)
    plt.scatter(np.arange(1), [-7.5], marker='*', color='red', s=50, label='Hamiltonian')
    plt.title('Kepler',fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=4, frameon=True)

    plt.subplot(133)
    K = np.load('./data/feature_data/koopman_84_0.03_kdv.npy')
    enc = np.load('./data/feature_data/enc_84_0.03_kdv.npy')
    print(K.shape, enc.shape)
    eigenvalue, featurevector = np.linalg.eig(K)
    print("特征值：", np.real(eigenvalue))
    print(featurevector.shape, enc.shape)
    print(featurevector[:, -1])
    weight = np.real(featurevector[:, -1])
    quantity = np.sum(weight * enc, axis=1)
    print(quantity.shape)
    var_list = []
    var_list.append(np.std(quantity) ** 2)
    for i in range(len(K)):
        var_list.append(np.std(enc[:, i]) ** 2)
    var_list = [np.log(x) for x in var_list]
    index = ['$g_c$', '$g_1$', '$g_2$', '$g_3$', '$g_4$', '$g_5$', '$g_6$', '$g_7$']
    color_list = ['k'] + ['gray' for _ in range(7)]
    bar_width = 0.5
    plt.bar(np.arange(len(index)) + bar_width * 0, var_list[:len(index)], color=color_list,
            width=bar_width)  # , color=seven_color[0], alpha=0.6, lw=2.0, edgecolor=seven_color[0],label='No Noise')
    plt.ylim(-17,0.5)
    plt.yticks([-17,0],fontsize=ticksize)
    plt.xticks(np.arange(len(index)) + bar_width * 0, index, fontsize=12)
    plt.xlabel('Component', fontsize=fontsize)
    plt.ylabel(r'$\log(\text{Temporal Variance})$', fontsize=fontsize, labelpad=-25)
    plt.scatter(np.arange(1), [-16.5], marker='*', color='red', s=50, label='Hamiltonian')
    plt.title('KdV',fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=4, frameon=True)

    plt.show()

# generate()
plot()
