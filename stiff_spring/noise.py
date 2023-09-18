import numpy as np

from functions import *

y = np.load('./data/true_y_12_200.npy')
noise = np.random.normal(0,1,100)
z = y[:,0]
def unify(x):
    x = x/np.max(x)
    mse = np.mean(np.abs(x[:,0]))
    return mse
x = np.array([np.random.uniform(-np.abs(i),np.abs(i),1) for i in z]).T[0]
print(x.shape)
plt.subplot(121)
plt.plot(np.arange(len(z)),z)
plt.scatter(np.arange(len(z)),x)
plt.subplot(122)
plt.scatter(x+z,y[:,1])
# plt.scatter(y[:,0],y[:,1])
err = unify(y)
print(err)
# plt.xlim(-15,15)
plt.show()
