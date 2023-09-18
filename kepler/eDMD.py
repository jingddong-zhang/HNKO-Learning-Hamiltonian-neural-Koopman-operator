import numpy as np

from functions import *

np.random.seed(369)
sigma = 0.03
y0 = torch.tensor([[1.0,0.0,0.0,0.9]])
t = torch.linspace(0, 5, 50)
true_y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
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
def dict_data(data):
    ord = 2 # order = ord-1
    x1,x2,x3,x4 = data[:,0],data[:,1],data[:,2],data[:,3]
    dict = []
    for i in range(ord):
        for j in range(ord):
            for k in range(ord):
                for l in range(ord):
                    dict.append(H(i,x1)*H(j,x2)*H(k,x3)*H(l,x4))
    return np.array(dict).T

# def dict_data(data):
#     ord = 4 # order = ord-1
#     x1,x2= data[:,0],data[:,1]
#     dict = []
#     for i in range(ord):
#         for j in range(ord):
#             dict.append(H(i,x1)*H(j,x2))
#     return np.array(dict).T
#
# def dict_data(data):
#     ord = 4 # order = ord-1
#     x1,x2= data[:,0],data[:,1]
#     dict = []
#     for i in range(ord):
#         for j in range(ord):
#             dict.append(H(i,x1)*H(j,x2))
#     return np.array(dict).T

def pred(K,data,n=300):
    # Y = np.zeros_like(data)
    Y = np.zeros([300,data.shape[1]])
    Y[0,:] = data[0,:]
    for i in range(len(Y)-1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]
    return Y

def decode(X,Y,pred):
    def mode(X,Y):
        # X: (n,4), Y: (n,p), B: (p,4)
        B = np.matmul(np.linalg.pinv(Y),X)
        return B
    B = mode(X,Y)
    pred_y = np.matmul(pred,B)
    return pred_y

lift_y = dict_data(y)
X1,X2 = lift_y[:-1].T,lift_y[1:].T
K = np.matmul(X2,np.linalg.pinv(X1))

Y = pred(K,dict_data(true_y))
pred_y = decode(y,lift_y,Y)
print(K.shape,pred_y.shape,np.max(np.abs(np.linalg.eigvals(K))))
# np.save('./data/true_0.03',true_y)
np.save('./data/edmd_300_0.03',pred_y)
plt.plot(true_y[:,0],true_y[:,1])
plt.plot(pred_y[:,0],pred_y[:,1])
plt.show()
