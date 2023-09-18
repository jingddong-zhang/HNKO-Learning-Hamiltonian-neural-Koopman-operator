import math
import numpy as np
import torch
from functions import *


class NC_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(NC_Net, self).__init__()
        self.recurrent_kernel = nn.Linear(n_input, n_output, bias=False)

    def forward(self, data):
        return self.recurrent_kernel(data)

class Encoder(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Encoder, self).__init__()

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

class Decoder(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
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

data = np.load('./data/samples_64_501.npy',allow_pickle=True).item()
t = data['t']
x = data['x']
true_y = data['u_x']
dim = true_y.shape[1]
k = np.max(np.sum(true_y**2,axis=1))
# plt.plot(t,np.sum(true_y**2,axis=1))
# plt.show()
np.random.seed(369)
sigma = 0.03
noise = np.random.normal(0,sigma,true_y.shape)
train_y = true_y + noise
'''
For learning 
'''
L = 50
T = 600

true_y = torch.from_numpy(true_y)
train_y = torch.from_numpy(train_y)

out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)

    model = NC_Net(dim+20,dim+20)
    H1 = int(1.2*dim)
    Enc = Encoder(dim, H1, dim+20)
    Dec = Decoder(dim+20, H1, dim)
    N = 100 # training data size
    i = 0
    max_iters = 5000
    learning_rate = 0.001
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in Enc.parameters()]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)
    while i < max_iters:
        # break
        lift_y = Enc(train_y)
        dec_y = Dec(lift_y)
        X1, X2, X3, X4 = lift_y[:N], lift_y[1:N+1], lift_y[2:N+2], lift_y[3:N+3]
        y1, y2, y3, y4 = train_y[:N], train_y[1:N+1], train_y[2:N+2], train_y[3:N+3]
        loss = torch.sum((X2 - model(X1)) ** 2) + torch.sum((X3 - model(model(X1))) ** 2) + torch.sum(
            (X4 - model(model(model(X1)))) ** 2) \
               + torch.sum((dec_y - train_y) ** 2) \
               + torch.sum((Dec(model(X1)) - y2) ** 2) + torch.sum((Dec(model(model(X1))) - y3) ** 2) + torch.sum(
            (Dec(model(model(model(X1)))) - y4) ** 2)

        print(out_iters,i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    print('\n')
    print("Total time: ", stop - start)

    K = model.recurrent_kernel.weight.detach().numpy()


    lift_y = Enc(true_y)
    pred_koop = Dec(torch.from_numpy(pred_data(K,lift_y.detach().numpy())))
    error = torch.sum((pred_koop - true_y) ** 2, dim=1)
    print('MSE:', error.mean())
    out_iters += 1


# plt.show()