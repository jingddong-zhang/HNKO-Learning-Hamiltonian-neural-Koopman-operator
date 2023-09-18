import math
import numpy as np
import torch
from functions import *


class lift_Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output,k):
        super(lift_Net, self).__init__()
        # torch.manual_seed(2)
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.Tanh(),
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

        self.k = torch.tensor(k, requires_grad=True)
        self.v = torch.randn([64+lift_d,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(64+lift_d, 120),
            nn.Tanh(),
            nn.Linear(120, 120),
            nn.Tanh(),
            nn.Linear(120,64)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)



data = np.load('./data/samples_64_501.npy',allow_pickle=True).item()
t = data['t']
x = data['x']
true_y = data['u_x']
dim = true_y.shape[1]
print(np.ptp(np.sum(true_y**2,axis=1)))
k = np.max(np.sum(true_y**2,axis=1))
# plt.plot(t,np.sum(true_y**2,axis=1))
# plt.show()
np.random.seed(369)
sigma = 0.03
noise = np.random.normal(0,sigma,true_y.shape)
train_y = true_y + noise
# X1, X2 = train_y[:300], train_y[1:301]
# X1 = X1.T
# X2 = X2.T
# K = np.matmul(X2, np.linalg.pinv(X1))
# pred_dmd = pred_data(K,true_y)


# plt.plot(x,pred_dmd[300,:])
# plt.show()

'''
For learning 
'''
L = 50
T = 600
lift_d = 20
q = 42
D_in = dim # input dimension
H1 = 10*D_in
D_out = dim # output dimension
train_y = torch.from_numpy(train_y)

out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    model = K_Net(D_in+lift_d, D_out+lift_d)
    g1 = lift_Net(64,120,64+lift_d,k)
    Dec = Decoder()

    i = 0
    max_iters = 5000
    learning_rate = 0.001
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in g1.parameters()]+[g1.k]+[g1.v]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)
    # optimizer = torch.optim.Adam([i for i in model.parameters()])
    while i < max_iters:
        # break
        lift_y = g1(train_y)
        dec_y = Dec(lift_y)
        X1,X2,X3,X4 = lift_y[:300],lift_y[1:301],lift_y[2:302],lift_y[3:303]
        v = g1.v/torch.sqrt(torch.sum(g1.v**2,dim=0))
        V = torch.mm(v.T,v)-torch.eye(q)
        # embed_y = torch.cat((y,g1(y)),dim=1)
        loss = torch.sum((X2-model(X1))**2) \
               +torch.sum((dec_y-train_y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                # +torch.sum(V**2)

        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i%2000==0:
            learning_rate = learning_rate*0.8
            optimizer = torch.optim.Adam(
                [i for i in model.parameters()] + [i for i in g1.parameters()] + [g1.k] + [g1.v] + \
                [i for i in Dec.parameters()], lr=learning_rate)
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()



    lift_y = g1(torch.from_numpy(true_y))
    # XX1,XX2 = re_data(true_y,lift_y)
    # dec_y = Dec(lift_y)[:-1]
    # y = y.detach().numpy()
    # dec_y=dec_y.detach().numpy()

    # Y = np.zeros([n,len(K)]) #NOKO
    # Y[0,:]=lift_y.detach().numpy()[0,:]
    # for i in range(n-1):
    #     x = Y[i, :].reshape(-1, 1)
    #     Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_koop = Dec(torch.from_numpy(pred_data(K,lift_y.detach().numpy()))).detach().numpy()
    np.save('./data/hkno_64_501_0.03', pred_koop)
    train_y = true_y+noise
    lift_y = g1(torch.from_numpy(train_y)).detach().numpy()
    X1, X2 = lift_y[:300], lift_y[1:301]
    X1 = X1.T
    X2 = X2.T
    K = np.matmul(X2,np.linalg.pinv(X1))
    pred_dmd = Dec(torch.from_numpy(pred_data(K,g1(torch.from_numpy(true_y)).detach().numpy()))).detach().numpy()
    plot(true_y,pred_dmd,pred_koop,L,T)
    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1


plt.show()