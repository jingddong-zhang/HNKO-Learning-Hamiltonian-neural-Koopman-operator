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

class K2_Net(torch.nn.Module):
    def __init__(self, n_input,n_output):
        super(K2_Net, self).__init__()
        # torch.manual_seed(2)
        self.layer1= nn.Linear(n_input, n_output, bias=False)
        self.layer2 = nn.Linear(n_input, n_output, bias=False)
        geotorch.orthogonal(self.layer1, "weight")
        geotorch.orthogonal(self.layer2, "weight")
        self.reset_parameters()

    def reset_parameters(self):
        # The manifold class is under `layer.parametrizations.tensor_name[0]`
        M1 = self.layer1.parametrizations.weight[0]
        M2 = self.layer2.parametrizations.weight[0]
        # Every manifold has a convenience sample method, but you can use your own initializer
        self.layer1.weight = M1.sample("uniform")
        self.layer2.weight = M2.sample("uniform")

    def forward(self, data):
        W1 = self.layer1.weight
        W2 = self.layer2.weight
        T = torch.kron(W1,W2)
        return torch.mm(data,T.T)
        # return self.recurrent_kernel(data)

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

        self.k = torch.tensor(5., requires_grad=True)
        # self.k = torch.tensor(5., requires_grad=True)
        self.v = torch.randn([12+lift_d,q], requires_grad=True)

    def forward(self, data):
        out = self.net(data)
        return out

class Decoder(nn.Module):

    def __init__(self,n_hidden):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(12+lift_d, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden,12)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        return self.net(y)




# np.random.seed(7)
np.random.seed(369)
y0 = torch.from_numpy(random_config()).view([1,12])
t = torch.linspace(0, 7, 200)
test_t = torch.linspace(0,15,300)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:]
test_y = odeint(threebody(), y0, test_t, atol=1e-8, rtol=1e-8)[:,0,:]
np.save('./data/true_chaos_7_200',true_y)
# X1,X2=y[:-1],y[1:]
print(torch.max(torch.sum(true_y**2,dim=1)))
# h = hami(true_y)
# plt.scatter(np.arange(len(h)),h)
# plt.ylim(-2,2)
# plt.scatter(true_y[:,0],true_y[:,1])
# plt.scatter(true_y[:,2],true_y[:,3])
# plt.scatter(true_y[:,4],true_y[:,5])
# plt.show()


'''
For learning 
'''
sigma = 0.01
lift_d = 400-12 # 19
q = 200 # 13
# lift_d = 100
# q = 56
D_in = 12+lift_d # input dimension
H1 = 120
D_out = 12+lift_d # output dimension


out_iters = 0
train_err = []
test_err = []
while out_iters < 1:
    break
    start = timeit.default_timer()
    torch.manual_seed(369)  # lift=0,q=6,seed=69
    np.random.seed(369)
    # model = K_Net(D_in, D_out)
    model = K2_Net(20, 20)
    y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))
    g1 = lift_Net(12,H1,12+lift_d)
    Dec = Decoder(H1)

    i = 0
    max_iters = 10000
    learning_rate = 0.0005
    optimizer = torch.optim.Adam([i for i in model.parameters()]+[i for i in g1.parameters()]+[g1.k]+[g1.v]+\
                                 [i for i in Dec.parameters()], lr=learning_rate)

    while i < max_iters:
        # break
        lift_y = g1(y)
        dec_y = Dec(lift_y)
        X1,X2,X3 = lift_y[:-2],lift_y[1:-1],lift_y[2:]
        v = g1.v/torch.sqrt(torch.sum(g1.v**2,dim=0))
        V = torch.mm(v.T,v)-torch.eye(q)
        # embed_y = torch.cat((y,g1(y)),dim=1)
        loss = torch.sum((X2-model(X1))**2)+ torch.sum((X3-model(model(X1)))**2)\
               +torch.sum((dec_y-y)**2) \
               +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
               +torch.sum(torch.mm(lift_y,v)**2)\
                +torch.sum(V**2)
        # loss = torch.sum((dec_y-y)**2) \
        #        +torch.sum((torch.sum(lift_y**2,dim=1)-g1.k)**2)\
        #        +torch.sum(torch.mm(lift_y,v)**2)\
        #         +torch.sum(V**2)
        # train_err.append(torch.mean(torch.sum((dec_y-y)**2,dim=1)))
        # test_err.append([torch.mean(torch.sum((Dec(g1(test_y[:5]))-test_y[:5])**2,dim=1)),torch.mean(torch.sum((Dec(g1(test_y[:20]))-test_y[:20])**2,dim=1)),
        #                 torch.mean(torch.sum((Dec(g1(test_y[:50]))-test_y[:50])**2,dim=1))])
        print(out_iters,i, "loss=", loss.item(),g1.k)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # if loss < 1.0:
        #     learning_rate = 0.0001
        #     optimizer = torch.optim.Adam(
        #         [i for i in model.parameters()] + [i for i in g1.parameters()] + [g1.k] + [g1.v] + \
        #         [i for i in Dec.parameters()], lr=learning_rate)
        if loss<=0.3:
            break
        i += 1
    stop = timeit.default_timer()

    # train_err = torch.tensor(train_err).detach().numpy()
    # test_err = torch.tensor(test_err).detach().numpy()
    # print(test_err.shape)
    # plt.plot(np.arange(len(train_err)),train_err,label='train err')
    # plt.plot(np.arange(len(train_err)), test_err[:,0],label='test err:5')
    # plt.plot(np.arange(len(train_err)), test_err[:, 1], label='test err:20')
    # plt.plot(np.arange(len(train_err)), test_err[:, 2], label='test err:50')
    # plt.legend()


    # model.load_state_dict(torch.load('./data/K.pkl'))
    # g1.load_state_dict(torch.load('./data/encoder.pkl'))
    # Dec.load_state_dict(torch.load('./data/decoder.pkl'))


    # j = 0
    # max_iters = 2000
    # learning_rate = 0.0001
    # optimizer = torch.optim.Adam([i for i in model.parameters()], lr=learning_rate)
    # while j < max_iters:
    #     lift_y = g1(y)
    #     dec_y = Dec(lift_y)
    #     X1,X2,X3 = lift_y[:-2],lift_y[1:-1],lift_y[2:]
    #     loss = torch.sum((X2 - model(X1)) ** 2) + torch.sum((X3 - model(model(X1))) ** 2)
    #     optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     print(j, "loss=", loss.item())
    #     j += 1

    # K = model.recurrent_kernel.weight.detach().numpy()
    K = torch.kron(model.layer1.weight,model.layer2.weight).detach().numpy()
    # X1 = X1.T.detach().numpy()
    # X2 = X2.T.detach().numpy()
    # K = np.matmul(X2,np.linalg.pinv(X1))

    # y0=torch.from_numpy(fix_config()).view([1,12])
    n = 200
    # t=torch.linspace(0,5*4,n)
    # true_y=odeint(threebody(),y0,t,atol=1e-8,rtol=1e-8)[:,0,:]
    y0.requires_grad_(True)
    t = torch.linspace(0, 7, 200)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :]
    lift_y = g1(true_y)
    # XX1,XX2 = re_data(true_y,lift_y)
    dec_y = Dec(lift_y)
    y = y.detach().numpy()
    dec_y=dec_y.detach().numpy()

    Y = np.zeros([n,len(K)]) #NOKO
    Y[0,:]=lift_y.detach().numpy()[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    pred_y = Dec(torch.from_numpy(Y)).detach().numpy()

    # m = 50
    # lift = lift_y[:-m]
    # for j in range(m):
    #     lift = model(lift)
    # pred_dec_y = Dec(lift).detach().numpy()

    np.save('./data/hnko_chaos_7_200_0.01',pred_y)
    Y = pred_y
    # plt.subplot(131)
    true_y = true_y.detach().numpy()
    # plt.plot(t.detach().numpy(),true_y[:,0])
    # plt.plot(t.detach().numpy(), dec_y[:, 0])
    # plt.axvline(10,ls='--')
    # plt.scatter(true_y[:,0],true_y[:,1])
    # plt.scatter(pred_y[:,0],pred_y[:,1])
    # plt.scatter(dec_y[:,0],dec_y[:,1])


    plt.plot(true_y[:,0],true_y[:,1])
    plt.plot(dec_y[:,0],dec_y[:,1])
    plt.plot(pred_y[:, 0], pred_y[:, 1])
    # plt.plot(pred_dec_y[:,0],pred_dec_y[:,1])
    # plt.subplot(132)
    # plt.scatter(true_y[:,2],true_y[:,3])
    # plt.scatter(pred_y[:,2],pred_y[:,3])

    # plt.plot(true_y[:,2],true_y[:,3])
    # plt.plot(dec_y[:,2],dec_y[:,3])
    # plt.plot(pred_y[:,2],pred_y[:,3])
    # plt.subplot(133)
    # plt.scatter(true_y[:,4],true_y[:,5])
    # plt.scatter(pred_y[:,4],pred_y[:,5])
    # plt.plot(true_y[:,4],true_y[:,5])
    # plt.plot(dec_y[:,2],dec_y[:,3])
    # plt.plot(pred_y[:,4],pred_y[:,5])
    # plot_4(true_y.detach().numpy(),pred_y,true_y.detach().numpy())
    # pred_pendulum(K,X1,XX1,50)
    # generate_plot(K,X1,XX1,49)
    # print('test err',torch.sum((X2 - model(X1))**2))
    print('\n')
    print("Total time: ", stop - start)
    out_iters += 1
    # torch.save(model.state_dict(),'./data/K.pkl')
    # torch.save(g1.state_dict(),'./data/encoder.pkl')
    # torch.save(Dec.state_dict(),'./data/decoder.pkl')




plt.show()