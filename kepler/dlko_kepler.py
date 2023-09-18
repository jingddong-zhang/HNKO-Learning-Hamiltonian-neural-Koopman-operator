import math

import matplotlib.pyplot as plt
import numpy as np

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

class NC_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(NC_Net, self).__init__()
        # torch.manual_seed(2)
        self.recurrent_kernel = nn.Linear(n_input, n_output, bias=False)

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

n = 50
# y0 = torch.tensor([[0.0,1.0,1.,0.0]])
y0 = torch.tensor([[1.0,0.0,0.,0.9]])
t = torch.linspace(0, 5, n)
# y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
true_y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]
# true_y = torch.cat((true_y[25:],true_y[:25]),dim=0)
# X1,X2=y[:-1],y[1:]
# print(torch.max(torch.sum(true_y**2,dim=1)))
# h = hami(y)
# print(np.max(h))
# plt.scatter(np.arange(len(h)),h)
# plt.ylim(-1,2)
# plt.scatter(true_y[:,0],true_y[:,1])
# plt.xlim(-1,1)
# plt.scatter(y[:,2],y[:,3])
# plt.show()

def re_data(y,lift_y):
    x = torch.cat((y,lift_y),dim=1)
    X1,X2=x[:-1],x[1:]
    return X1,X2

def pred_pendulum(K,X,T,n):
    Y = np.zeros([n,len(K)])
    X = X.detach().numpy()
    T = T.detach().numpy()
    Y[0,:]=T[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    plt.subplot(121)
    plt.scatter(np.arange(len(k_energy(T))),k_energy(T),label='kinetic')
    plt.scatter(np.arange(len(p_energy(T))),p_energy(T),label='total')
    plt.scatter(np.arange(len(t_energy(T))),t_energy(T),label='potential')
    plt.title('true energy')
    # plt.scatter(T[:, 0], T[:, 1], label='true')
    # plt.scatter(T[:,2],T[:,3],label='true')
    # plt.scatter(X[:,0],X[:,1],label='noisy')
    # plt.scatter(X[:,2], X[:, 3], label='noisy')
    # plt.scatter(Y[:,0],Y[:,1],label='pred')
    plt.legend()
    plt.subplot(122)
    plt.scatter(np.arange(len(k_energy(Y))),k_energy(Y),label='kinetic')
    plt.scatter(np.arange(len(p_energy(Y))),p_energy(Y),label='total')
    plt.scatter(np.arange(len(t_energy(Y))),t_energy(Y),label='potential')
    plt.title('pred energy')
    # plt.scatter(Y[:, 0], Y[:, 1], label='pred')
    # plt.xlim(-1.5,1.5)
    # plt.title(r'$\sigma={},dim~lift={}$'.format(sigma, lift_d),fontsize=15)
    plt.legend()
    plt.show()


def visual(x):
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x[:,0],x[:,1],x[:,2])
    plt.show()

# visual(lift(y).detach().numpy())



'''
For learning 
'''
sigma = 0.03
lift_d = 20
q = 13
D_in = 4+lift_d # input dimension
H1 = 10*D_in
D_out = 4+lift_d # output dimension


out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(69)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    Enc = lift_Net(4,64,4+lift_d)
    Dec = Decoder(64)
    # g1.load_state_dict(torch.load('./data/lift_0.pkl'))
    # lift_y = g1(y)
    # X1,X2 = re_data(y,lift_y)

    # visual(X1.detach().numpy())
    i = 0
    max_iters = 10000
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in Enc.parameters()]+
                                 [i for i in Dec.parameters()], lr=learning_rate)
    # optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
    # optimizer = torch.optim.Adam([i for i in g1.parameters()]+[g1.k]+[g1.v], lr=learning_rate)
    while i < max_iters:
        # break
        lift_y = Enc(y)
        dec_y = Dec(lift_y)
        X1,X2,X3,X4 = lift_y[:47],lift_y[1:48],lift_y[2:49],lift_y[3:50]
        y1, y2, y3, y4 = y[:47], y[1:48], y[2:49], y[3:50]
        loss = torch.sum((X2-model(X1))**2)+torch.sum((X3-model(model(X1)))**2)+torch.sum((X4-model(model(model(X1))))**2) \
               +torch.sum((dec_y-y)**2) \
                +torch.sum((Dec(model(X1))-y2)**2)+torch.sum((Dec(model(model(X1)))-y3)**2)+torch.sum((Dec(model(model(model(X1))))-y4)**2)

        print(out_iters, i, "loss=", loss.item())
        # print(out_iters,i, "loss=", loss.item(),g1.k,g1.v[:3])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-3:
            break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()
    # X1 = X1.T.detach().numpy()
    # X2 = X2.T.detach().numpy()
    # K = np.matmul(X2,np.linalg.pinv(X1))

    lift_y = Enc(true_y)
    # XX1,XX2 = re_data(true_y,lift_y)
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
    np.save('./data/nc_300_0.03',pred_y)
    Y = pred_y
    # plt.plot(true_y[:,0],true_y[:,1])
    # plt.plot(dec_y[:,0],dec_y[:,1])
    # plt.plot(pred_y[:,0],pred_y[:,1])
    # plot_4(true_y.detach().numpy(),pred_y,true_y.detach().numpy())
    plt.subplot(121)
    plt.scatter(np.arange(len(k_energy(Y))),k_energy(Y),label='kinetic')
    plt.scatter(np.arange(len(p_energy(Y))),p_energy(Y),label='total')
    plt.scatter(np.arange(len(t_energy(Y))),t_energy(Y),label='potential')
    plt.subplot(122)
    Y = np.load('./data/noko_0.03.npy')
    plt.scatter(np.arange(len(k_energy(Y))),k_energy(Y),label='kinetic')
    plt.scatter(np.arange(len(p_energy(Y))),p_energy(Y),label='total')
    plt.scatter(np.arange(len(t_energy(Y))),t_energy(Y),label='potential')
    # pred_pendulum(K,X1,XX1,50)
    # generate_plot(K,X1,XX1,49)
    # print('test err',torch.sum((X2 - model(X1))**2))
    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1


plt.show()