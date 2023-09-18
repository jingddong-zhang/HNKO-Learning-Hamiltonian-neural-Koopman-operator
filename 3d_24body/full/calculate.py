import ot
import numpy as np
import matplotlib.pyplot as plt

#
a = np.load('./data/3d_24body_full.npy')
b = np.load('./data/hnko_154_80_0.03_64.npy')
c = np.load('./data/dlko_0.03_64_15000.npy')
d = np.load('./data/dmd_0.03.npy')
f = np.load('./data/sympnet_24body_0.03_G_128_10000_20.npy')
g = np.load('./data/hnn_0.03_3_64.npy')

def pW_cal(a, b, p=2, metric='euclidean'):
    """ Args:
            a, b: samples sets drawn from α,β respectively
            p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
            metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
    """
    # cost matrix
    M = ot.dist(a, b, metric=metric)

    M = pow(M, p)

    # uniform distribution assumption
    alpha = ot.unif(len(a))
    beta = ot.unif(len(b))

    # p-Wasserstein Distance
    pW = ot.emd2(alpha, beta, M, numItermax=500000)
    pW = pow(pW, 1 / p)

    return pW

def LTSE_cal(a, b):
    error = np.sqrt(np.sum((a-b)**2,axis=1))
    error = error.mean()
    return error


print('HNKO:','Wass-2 is {}'.format(pW_cal(a,b)),'LTSE is {}'.format(LTSE_cal(a,b)))
print('DLKO:','Wass-2 is {}'.format(pW_cal(a,c)),'LTSE is {}'.format(LTSE_cal(a,c)))
print('DMD:','Wass-2 is {}'.format(pW_cal(a,d)),'LTSE is {}'.format(LTSE_cal(a,d)))
print('G-SYm:','Wass-2 is {}'.format(pW_cal(a,f)),'LTSE is {}'.format(LTSE_cal(a,f)))
print('HNN:','Wass-2 is {}'.format(pW_cal(a,g)),'LTSE is {}'.format(LTSE_cal(a,g)))