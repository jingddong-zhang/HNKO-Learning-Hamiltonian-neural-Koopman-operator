import math
import numpy as np
import torch
from functions import *




# data = np.load('./data/samples_64_501.npy',allow_pickle=True).item()
data = np.load('./data/samples_1024_501.npy',allow_pickle=True).item()
t = data['t']
true_y = data['u_x']
# true_y = np.load('./data/sol_256.npy')
dim = true_y.shape[1]
# plt.plot(t,np.sum(true_y**2,axis=1))
# plt.show()
np.random.seed(369)
sigma = 0.03
noise = np.random.normal(0,sigma,true_y.shape)
train_y = true_y + noise

# X1, X2 = train_y[:100], train_y[1:101]
# X1 = X1.T
# X2 = X2.T
# K = np.matmul(X2, np.linalg.pinv(X1))
# pred_dmd = pred_data(K, true_y)
# error = np.sum((true_y - pred_dmd)**2, axis=1)
# plt.plot(np.arange(len(error)),error)
# print('MSE:', error[400:].mean())

def dmd(train_y,true_y):
    X1, X2 = train_y[:300], train_y[1:301]
    X1 = X1.T
    X2 = X2.T
    K = np.matmul(X2,np.linalg.pinv(X1))
    pred_y = pred_data(K,true_y)
    np.save('./data/dmd_64_501_0.03',pred_y)
# dmd(train_y,true_y)

'''
For learning 
'''
L = 50
T = 600
D_in = dim # input dimension
D_out = dim # output dimension
train_y = torch.from_numpy(train_y)

out_iters = 0
eigen_list=[]
while out_iters < 1:
    # break
    start = timeit.default_timer()
    torch.manual_seed(369)
    model = K_Net(D_in, D_out)
    # model = K2_Net(32,32)
    i = 0
    max_iters = 5000
    learning_rate = 0.001
    optimizer = torch.optim.Adam([i for i in model.parameters()])
    while i < max_iters:
        # break
        N = 100
        X1,X2,X3,X4 = train_y[:N],train_y[1:N+1],train_y[2:N+2],train_y[3:N+3]
        loss = torch.sum((X2-model(X1))**2)+torch.sum((X3-model(model(X1)))**2)+torch.sum((X4-model(model(model(X1))))**2)
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
    # K = torch.kron(model.layer1.weight, model.layer2.weight).detach().numpy()
    K = model.recurrent_kernel.weight.detach().numpy()
    pred_koop = pred_data(K,true_y)
    # np.save('./data/hkno_64_501_0.03',pred_koop)
    train_y = true_y+noise
    X1, X2 = train_y[:N], train_y[1:N+1]
    X1 = X1.T
    X2 = X2.T
    K = np.matmul(X2,np.linalg.pinv(X1))
    pred_dmd = pred_data(K,true_y)
    plot(true_y,pred_dmd,pred_koop,L,T)
    # print('test err',torch.sum((X2 - model(X1))**2))

    # torch.save(model.state_dict(), './data/inv_eigen_{}.pkl'.format(out_iters))
    # torch.save(g1.state_dict(), './data/lift_{}.pkl'.format(out_iters)) # 10000,0.01,369
    out_iters += 1


plt.show()