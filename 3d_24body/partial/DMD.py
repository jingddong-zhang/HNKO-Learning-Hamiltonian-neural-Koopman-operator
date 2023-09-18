import numpy as np

true_y = np.load('./data/3d_24body.npy')
np.random.seed(369)
sigma = 0.03
y = true_y + np.random.normal(0, sigma, true_y.shape)


X1,X2 = y[:-1],y[1:]
X1 = X1.T
X2 = X2.T
K = np.matmul(X2,np.linalg.pinv(X1))

Y = np.zeros([len(y), len(K)])  # NOKO
Y[0, :] = true_y[0, :]
for i in range(len(y)-1):
    x = Y[i, :].reshape(-1, 1)
    Y[i + 1, :] = np.matmul(K, x).T[0]

np.save('./data/dmd_{}'.format(sigma),Y)