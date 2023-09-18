import numpy as np
import torch

from functions import *

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
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)
        self.k = torch.tensor(6000.0, requires_grad=True)
        self.v = torch.tensor([1.0,1.0,1.0], requires_grad=True)

    def forward(self, data):
        # sigmoid = torch.nn.ReLU()
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(data))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

class AutoEncoder(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(AutoEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_output)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)

y0 = torch.tensor([[1.0,0.0]])
t = torch.linspace(0, 3, 100)
# y = odeint(pendulum(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
stiff_list = torch.linspace(1.0,200,40)
a = stiff_list[0]
k = torch.pow(a,1)
m = torch.pow(a,1)
func = spring(k,m)
# y = odeint(func, y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
# X1,X2=y[:-1],y[1:]
#
# plt.subplot(121)
# plt.plot(y[:,0],y[:,1])
# plt.subplot(122)
# plt.plot(y[:,0],y[:,1])
# # plt.xlim(-60,60)
# plt.show()


def re_data(y,lift_y):
    x = torch.cat((y,lift_y),dim=1)
    X1,X2=x[:-1],x[1:]
    return X1,X2

def pred_pendulum(K,X,n):
    Y = np.zeros([n,len(K)])
    X = X.detach().numpy()
    Y[0,:]=X[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    # plt.subplot(121)
    # plt.scatter(X[:,0],X[:,1],label='noisy')
    # plt.subplot(122)
    # plt.scatter(Y[:,0],Y[:,1],label='pred')
    # # plt.xlim(-1.5,1.5)
    # plt.legend()
    # plt.show()
    return Y

def lift(y):
    Z = torch.zeros([len(y),3])
    # r = torch.max(y)**2+torch.min(torch.abs(y))**2
    r=torch.max(torch.sum(y**2,dim=1))
    for i in range(len(Z)):
        Z[i,:2]=y[i,:]
        z = torch.sqrt(r-torch.sum(y[i,:]**2))
        if y[i,0]>=0:
            Z[i,2]=z
        else:
            Z[i,2]=-z
    return Z


def visual(x):
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x[:,0],x[:,1],x[:,2])
    plt.show()

# print(lift(y))
# visual(lift(y).detach().numpy())

def dlko():
    train_data = np.load('./data/train_y_3_100.npy')
    # y0 = torch.tensor([[1.0, 0.0]])
    # t = torch.linspace(0, 10, 100)
    pred_data = np.zeros([40,300,2]) # save prediction data
    out_iters = 0
    while out_iters < 40:
        start = timeit.default_timer()
        model = NC_Net(5, 5)
        Enc = AutoEncoder(2,20,5)
        Dec = AutoEncoder(5,20,2)
        true_y = torch.from_numpy(train_data[out_iters,:,:2])
        np.random.seed(369)
        y = true_y + torch.from_numpy(np.random.normal(0,0.03,true_y.shape))
        # y = y.requires_grad_(True)
        # y = lift(y)
        # X1, X2 = y[:-1], y[1:]
        i = 0
        max_iters = 3000
        learning_rate = 0.05
        optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
        while i < max_iters:
            # break
            lift_y = Enc(y)
            dec_y = Dec(lift_y)
            X1, X2, X3, X4 = lift_y[:97], lift_y[1:98], lift_y[2:99], lift_y[3:100]
            y1, y2, y3, y4 = y[:97], y[1:98], y[2:99], y[3:100]
            loss = torch.sum((X2 - model(X1)) ** 2) + torch.sum((X3 - model(model(X1))) ** 2) + torch.sum(
                (X4 - model(model(model(X1)))) ** 2) \
                   + torch.sum((dec_y - y) ** 2) \
                   + torch.sum((Dec(model(X1)) - y2) ** 2) + torch.sum((Dec(model(model(X1))) - y3) ** 2) + torch.sum(
                (Dec(model(model(model(X1)))) - y4) ** 2)

            # loss = torch.sum((X2 - model(X1)) ** 2)
            print(out_iters, i, "loss=", loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if loss <= 1e-4:
                break
            i += 1
        stop = timeit.default_timer()
        K = model.recurrent_kernel.weight.detach().numpy()
        y0 = torch.tensor([[1.0, 0.0]])
        # lift_y = Enc(y0)
        lift_y = Enc(y)

        n = 300
        Y = np.zeros([n, len(K)])  # NC:DLKO
        Y[0, :] = lift_y.detach().numpy()[0, :]
        for i in range(n - 1):
            x = Y[i, :].reshape(-1, 1)
            Y[i + 1, :] = np.matmul(K, x).T[0]
        pred_y = Dec(torch.from_numpy(Y)).detach().numpy()

        # pred_koop = pred_pendulum(K, X1, 300)
        pred_data[out_iters, :] = pred_y[:,:2]
        print('\n')
        print("Total time: ", stop - start)
        out_iters += 1
    np.save('./data/nc_0.03_9_300', pred_data)

dlko()

