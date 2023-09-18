import numpy as np

from functions import *

np.random.seed(369)
sigma = 0.03
y0=torch.from_numpy(fix_config()).view([1,12])
t = torch.linspace(0, 5, 100)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:,0,:].detach().numpy()
y = true_y+np.random.normal(0,sigma,true_y.shape)

def H(i,x):
    def H0(x):
        return np.zeros_like(x)+1
    def H1(x):
        return 2*x
    def H2(x):
        return 4*x**2-2
    def H3(x):
        return 8*x**3-12*x
    def H4(x):
        return 16*x**4-48*x**2+12
    if i==0:
        return H0(x)
    if i==1:
        return H1(x)
    if i==2:
        return H2(x)
    if i==3:
        return H3(x)
    if i==4:
        return H4(x)

# def dict_data(data):
#     ord = 2 # order = ord-1
#     x1,x2,x3,x4,x5,x6 = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]
#     x7,x8,x9,x10,x11,x12 = data[:,6],data[:,7],data[:,8],data[:,9],data[:,10],data[:,11]
#     dict = []
#     for k1 in range(ord):
#         for k2 in range(ord):
#             for k3 in range(ord):
#                 for k4 in range(ord):
#                     for k5 in range(ord):
#                         for k6 in range(ord):
#                             for k7 in range(ord):
#                                 for k8 in range(ord):
#                                     for k9 in range(ord):
#                                         for k10 in range(ord):
#                                             for k11 in range(ord):
#                                                 for k12 in range(ord):
#                                                     dict.append(H(k1,x1)*H(k2,x2)*H(k3,x3)*H(k4,x4)*H(k5,x5)*H(k6,x6)*H(k7,x7)\
#                                                                 *H(k8,x8)*H(k9,x9)*H(k10,x10)*H(k11,x11)*H(k12,x12))
#                                                     print(k1,k3,k5,k7,k9,k11)
#     return np.array(dict).T

def dict_data(data):
    ord = 2 # order = ord-1
    x1,x2,x3,x4,x5,x6 = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]
    dict = []
    for k1 in range(ord):
        for k2 in range(ord):
            for k3 in range(ord):
                for k4 in range(ord):
                    for k5 in range(ord):
                        for k6 in range(ord):
                            dict.append(H(k1,x1)*H(k2,x2)*H(k3,x3)*H(k4,x4)*H(k5,x5)*H(k6,x6))
    return np.array(dict).T

def pred(K,data,n):
    Y = np.zeros([n,data.shape[1]])
    Y[0,:] = data[0,:]
    for i in range(n-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    return Y

def decode(X,Y):
    def mode(X,Y):
        # X: (n,12), Y: (n,p), B: (p,12)
        B = np.matmul(np.linalg.pinv(Y),X)
        return B
    B = mode(X,Y)
    # pred_y = np.matmul(Y,B)
    # return pred_y
    return B

start = timeit.default_timer()
# print(y.shape)
lift_y = dict_data(y)
print(lift_y.shape)
X1,X2 = lift_y[:-1].T,lift_y[1:].T
K = np.matmul(X2,np.linalg.pinv(X1))
dec = decode(y,lift_y)
# print(dict_data(true_y).shape[1],dec.shape)
Y = pred(K,dict_data(true_y),100)
pred_y = np.matmul(Y,dec)
# pred_y = decode(y,lift_y)
print(K.shape,Y.shape,pred_y.shape)
# np.save('./data/true_0.03',true_y)
# np.save('./data/he_edmd_90_1800_0.03',pred_y)

end = timeit.default_timer()
print(end-start)


# plt.plot(true_y[:,0],true_y[:,1])
plt.plot(pred_y[:,0],pred_y[:,1])
plt.show()
