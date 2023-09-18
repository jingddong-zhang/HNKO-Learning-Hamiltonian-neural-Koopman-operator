import math

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
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,n_hidden),
            nn.Tanh(),
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

        self.k = torch.tensor(2.0, requires_grad=True)
        self.v = torch.randn([12+lift_d,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(12+lift_d, 120),
            nn.Tanh(),
            nn.Linear(120, 120),
            nn.Tanh(),
            nn.Linear(120,12)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)




# np.random.seed(7)
y0 = torch.from_numpy(fix_config(0.9)).view([1,12])
# y00 = torch.from_numpy(fix_config(0.6)).view([1,12])

print(y0)
t = torch.linspace(0, 7.05, 100)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]
# true_yy = odeint(threebody(), y00, t, atol=1e-8, rtol=1e-8)[:,0,:]
# X1,X2=y[:-1],y[1:]
# np.save('./data/vary_true_400',test_y)
print(torch.max(torch.sum(true_y**2,dim=1)))
# h = hami(true_y)
# plt.scatter(np.arange(len(h)),h)
# plt.ylim(-2,2)
# plt.plot(true_y[:,0],true_y[:,1])
# plt.plot(true_yy[:,0],true_yy[:,1])
# plt.scatter(true_y[:,2],true_y[:,3])
# plt.scatter(true_y[:,4],true_y[:,5])
# plt.show()


'''
For learning 
'''
sigma = 0.0
lift_d = 19 # 19
q = 15  # 13
D_in = 12+lift_d # input dimension
H1 = 10*D_in
D_out = 12+lift_d # output dimension

# vary_initial = np.zeros([10,20,400,12])
out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    # true_y = True_y[:(out_iters+1)*10]
    # true_y = True_y
    print('data size:',len(true_y))
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    model = K_Net(D_in, D_out)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    # y = y[:(out_iters+1)*5]
    g1 = lift_Net(12,120,12+lift_d)
    Dec = Decoder()

    i = 0
    max_iters = 3000
    learning_rate = 0.001
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in g1.parameters()]+[g1.k]+[g1.v]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)
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
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2) \
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)
        print(out_iters,i, "loss=", loss.item(),g1.k)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    stop = timeit.default_timer()
    K = model.recurrent_kernel.weight.detach().numpy()

    # y0=torch.from_numpy(fix_config()).view([1,12])
    np.random.seed(5)
    y0 = torch.from_numpy(select_fix_config(np.random.normal(0, 20, 2), np.random.uniform(0.7, 0.8, 1))).view([1, 12])
    n = 400
    t=torch.linspace(0,7.05*4,n)
    true_y=odeint(threebody(),y0,t,atol=1e-8,rtol=1e-8)[:,0,:]

    lift_y = g1(true_y)
    # XX1,XX2 = re_data(true_y,lift_y)
    dec_y = Dec(lift_y)[:-1]
    y = y.detach().numpy()
    dec_y=dec_y.detach().numpy()

    Y = np.zeros([n,len(K)]) #NOKO
    Y[0,:]=lift_y.detach().numpy()[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()
    # vary_initial[k,out_iters,:,:] = pred_y

    # np.save('./data/hnko_31_15_90_1800_0.03',pred_y)
    Y = pred_y
    # plt.subplot(131)
    plt.plot(true_y[:,0],true_y[:,1])
    # plt.plot(dec_y[:,0],dec_y[:,1])
    plt.plot(pred_y[:,0],pred_y[:,1])
    # plt.subplot(132)
    # plt.plot(true_y[:,2],true_y[:,3])
    # plt.plot(dec_y[:,2],dec_y[:,3])
    # plt.plot(pred_y[:,2],pred_y[:,3])
    print('\n')
    print("Total time: ", stop - start)
    out_iters += 1

plt.show()