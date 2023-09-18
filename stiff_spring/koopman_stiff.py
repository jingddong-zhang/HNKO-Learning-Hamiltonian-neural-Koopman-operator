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

def hkno():
    train_data = np.load('./data/train_y_3_100.npy')
    # y0 = torch.tensor([[1.0, 0.0]])
    # t = torch.linspace(0, 10, 100)
    pred_data = np.zeros([40,300,2]) # save prediction data
    out_iters = 0
    while out_iters < 40:
        start = timeit.default_timer()
        model = K_Net(3, 3)
        y = torch.from_numpy(train_data[out_iters,:,:2])
        np.random.seed(369)
        y += torch.from_numpy(np.random.normal(0,0.03,y.shape))
        # y = y.requires_grad_(True)
        y = lift(y)
        X1, X2 = y[:-1], y[1:]
        i = 0
        max_iters = 3000
        learning_rate = 0.05
        optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
        while i < max_iters:
            # break
            loss = torch.sum((X2 - model(X1)) ** 2)
            print(out_iters, i, "loss=", loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if loss <= 1e-4:
                break
            i += 1
        stop = timeit.default_timer()
        K = model.recurrent_kernel.weight.detach().numpy()
        pred_koop = pred_pendulum(K, X1, 300)
        pred_data[out_iters, :] = pred_koop[:,:2]
        print('\n')
        print("Total time: ", stop - start)
        out_iters += 1
    # np.save('./data/hkno_9_300',pred_data)
    np.save('./data/hkno_0.03_9_300', pred_data)

hkno()
'''
For learning 
'''
D_in = 3 # input dimension
H1 = 10*D_in
D_out = 3 # output dimension

out_iters = 0
eigen_list=[]
while out_iters < 1:
    break
    start = timeit.default_timer()
    # torch.manual_seed(out_iters*9)
    # np.random.seed(out_iters)
    model = K_Net(D_in, D_out)
    y = lift(y)
    X1, X2 = y[:-1], y[1:]
    # y = y + torch.from_numpy(np.random.normal(0,0.05,y.shape))
    # XX1,XX2=y[:-1],y[1:]
    # g1 = lift_Net(2,20,1)
    # g1.load_state_dict(torch.load('./data/lift_0.pkl'))
    # lift_y = g1(y)
    # X1,X2 = re_data(y,lift_y)
    # visual(X1.detach().numpy())
    i = 0
    max_iters = 2000
    learning_rate = 0.05
    optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
    # optimizer = torch.optim.Adam([i for i in g1.parameters()]+[g1.k]+[g1.v], lr=learning_rate)
    while i < max_iters:
        # break
        loss = torch.sum((X2-model(X1))**2)
        # v = g1.v/torch.sqrt(torch.sum(g1.v**2))
        # loss = torch.sum((torch.sum(y**2,dim=1)+g1(y).T[0]**2-g1.k)**2)\
        #        +torch.sum((torch.sum(v*torch.cat((y,g1(y)),dim=1),dim=1))**2)
        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()
    pred_koop=pred_pendulum(K,X1,200)
    # np.save('./data/hkno_12_200',pred_koop)
    # print('test err',torch.sum((X2 - model(X1))**2))
    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters))
    out_iters += 1

