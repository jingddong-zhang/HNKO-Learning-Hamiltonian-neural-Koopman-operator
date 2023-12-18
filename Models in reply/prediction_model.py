# 利用RC， 对Lorenz系统拟合
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
from my_function import get_lorenz_data, initial_W_in, \
    load_matrix_A, get_R, get_X, get_P, Predict_V, \
    plot_for_creat_model, get_next_True_Pred, plot_True_Pred, generate_file,get_next_Pred,generate_A
import time
from functions import *
################################################################################
# 参数， 具体见Jaideep Pathak PRL 2017 附录
D_in = 12   # 输入维度
D_out = D_in  # 输出等于输出维度

sigma = 1.0   # 输入矩阵的元素按均匀分布【-sigma, sigma】采样
rho = 0.6     # 谱半径


# 输出结果文件
out_file = './out_fig'
generate_file(path=out_file)


np.random.seed(369)

num = 100
y0=torch.from_numpy(fix_config()).view([1,12])
t = torch.linspace(0, 5, 100)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:,0,:]
y = true_y + np.random.normal(0,0.03,true_y.shape)
All_U = y.T
print('U shape: ', All_U.shape)

################################################################################
################-------模型搭建与拟合---------################
# 按PRL 2017 生成输入矩阵W_in, 具体看文章或相应的函数



TT = [num-1]
T = num - 1
D_r_list = [360,960,1440,2400]
rho_list = [0.6,0.9,1.2]
p_list = [0.3,0.6,0.9]
Length = 1700  # 预测训练数据接下来的50个步长数据
figsize = (18, 9)
fontsize = 18

def plot_prediction_fig():
    start = time.time()
    for i, D_r in enumerate(D_r_list):
        for j, rho in enumerate(rho_list):
            for k, p in enumerate(p_list):
                A, max_eigval = generate_A(shape=(D_r, D_r), rho=rho, D_r=D_r, p=p)
                W_in = initial_W_in([D_r, D_in], -sigma, sigma)
                U = All_U[:, -T:]  # 利用最后面T个步长的时间序列数据作为训练数据

                # 根据PRL 2017的附录中的方法，求解

                R = get_R(A, W_in, U, T)
                X = get_X(R)
                P, P1, P2 = get_P(U, X, beta)
                V = np.matmul(P1, R) + np.matmul(P2, R**2)

                # 得到训练后输出层的参数P, P1, P2，进行预测训练数据接下来的Length个步长数据
                pred_y = get_next_Pred(Length, V, R, P1, P2, A, W_in).T

                print(pred_y.shape)
                np.save('./data/RC 3body D_r={} rho={} p={}'.format(D_r,rho,p),pred_y)
                print('({},{},{})'.format(i,j,k),'time used: %.2f s' % (time.time() - start))
    end = time.time()
    print('用时 %.2fs' % (end - start))


def RC_different_data_length():
    np.random.seed(369)
    beta = 1e-2  # 正则化系数

    start = time.time()
    D_r = 1440
    p = 0.9
    rho = 0.6
    # for i, var in enumerate([0.0,0.01,0.02,0.03]):
    for i, num in enumerate([100]):
        # num = 100
        var = 0.05
        T = num - 1
        y0 = torch.from_numpy(fix_config()).view([1, 12])
        t = torch.linspace(0, 5, num)
        true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:, 0, :]
        y = true_y + np.random.normal(0, var, true_y.shape)
        All_U = y.T
        print('U shape: ', All_U.shape)

        A, max_eigval = generate_A(shape=(D_r, D_r), rho=rho, D_r=D_r, p=p)
        W_in = initial_W_in([D_r, D_in], -sigma, sigma)
        U = All_U[:, -T:]  # 利用最后面T个步长的时间序列数据作为训练数据

        # 根据PRL 2017的附录中的方法，求解

        R = get_R(A, W_in, U, T)
        X = get_X(R)
        P, P1, P2 = get_P(U, X, beta)
        V = np.matmul(P1, R) + np.matmul(P2, R**2)

        # 得到训练后输出层的参数P, P1, P2，进行预测训练数据接下来的Length个步长数据
        pred_y = get_next_Pred(Length, V, R, P1, P2, A, W_in).T

        print(pred_y.shape)
        np.save('./data/RC 3body num={} sigma={} beta={}'.format(num,var,beta),pred_y)
        print('({})'.format(num),'time used: %.2f s' % (time.time() - start))
    end = time.time()
    print('用时 %.2fs' % (end - start))

def calculate():
    y0 = torch.from_numpy(fix_config()).view([1, 12])
    true = odeint(threebody(), y0, torch.linspace(0, 90, 1800), atol=1e-8, rtol=1e-8)[300:1200,0,:].detach().numpy()
    error_list = []
    for D_r in D_r_list:
        for rho in rho_list:
            for p in p_list:
                pred = np.load('./data/RC_data/RC 3body D_r={} rho={} p={}.npy'.format(D_r, rho, p))[200:1100]
                error = np.sqrt(np.sum((true-pred)**2,axis=1)).mean()
                # print(layer,lr,H1,int(error*1000)/1000)
                print(D_r, rho, p, '{:.2f}'.format(error))
                error_list.append(error)
    print(min(error_list))
    for i,_ in enumerate(error_list):
        if i%9 == 0:
            print('\multirow{4}{*}{}',end=' ')
            print('& $D_r= $')
        print('&${:.2f}$'.format(_),end=' ')
        if (i+1)%9==0:
            print('\\\\')
            print('\n')
if __name__ == '__main__':
    # plot_prediction_fig()  # 预测和真实的曲线图
    # calculate()
    RC_different_data_length()