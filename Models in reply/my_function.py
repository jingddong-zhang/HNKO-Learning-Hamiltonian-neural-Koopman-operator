import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph
from scipy.integrate import odeint
import os

def generate_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(path):
    W_in = np.load(path + '/' + 'W_in.npy')
    A = np.load(path + '/' + 'A.npy')
    P1 = np.load(path + '/' + 'P1.npy')
    P2 = np.load(path + '/' + 'P2.npy')
    R = np.load(path + '/' + 'R.npy')
    V = np.load(path + '/' + 'V.npy')
    return W_in, A, P1, P2, R, V


def load_matrix_A(path):
    return np.load(path)


def get_lorenz_data(state0, t):
    state0 = state0.flatten()
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    def f(state, t):
        x, y, z = state  # unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

    states = odeint(f, state0, t)
    return states.transpose()


def initial_W(shape, low_bound, up_bound):
    return np.random.uniform(low_bound, up_bound, size=shape)


def initial_W_in(shape, low_bound, up_bound):
    # 按附录，PRL Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach
    # 设置输入权重W_{in}的权重
    '''
    The elements of Win are chosen so that every node in the network receives exactly one scalar input from u(t)
    while each scalar input in u(t) is connected to Dr/Din nodes in the network. The non-zero elements of Win
    are chosen randomly from a uniform distribution in [-sigma, sigma]
    '''
    W = np.zeros(shape)
    index1 = np.arange(shape[0])
    index2 = np.tile(np.arange(shape[1]), int(shape[0] / shape[1]))
    W[index1, index2] = initial_W([shape[0], ], low_bound, up_bound)
    return W


# 生成RC的recurrent的随机矩阵A, R(t+1)=tanh(AR(t)+Bu(t))
def generate_A(shape, rho, D_r,p):
    '''
    :param shape: 矩阵A的shape  （D_r, D_r）
    :param rho:   矩阵A的谱半径
    :param D_r:   矩阵A的维度
    :return:  生成后的矩阵A
    '''
    G = erdos_renyi_graph(D_r, p, seed=1)   # 生成ER图，节点为D_r个，连接概率p = 3 /D_r
    degree = [val for (node, val) in G.degree()]
    print('平均度:', sum(degree) / len(degree))
    G_A = nx.to_numpy_matrix(G)  # 生成后的图转化为邻接矩阵A， 有边的为1，无边为0
    index = np.where(G_A > 0)  # 找出有边的位置
    
    res_A = np.zeros(shape)

    a = 0.3
    res_A[index] = initial_W([len(index[0]), ], -1.0, 1.0)  # 对有边的位置按均匀分布[0, a]进行随机赋值, [0,0.3]
    max_eigvalue = np.real(np.max(LA.eigvals(res_A)))  # 计算生成后矩阵A的最大特征值
    print('before max_eigvalue:{}'.format(max_eigvalue))
    res_A = res_A / abs(max_eigvalue) * rho  # 调整矩阵A的谱半径，使得A的谱半径为预先设定的rho
    max_eigvalue = np.real(np.max(LA.eigvals(res_A)))
    print('after max_eigvalue:{}'.format(max_eigvalue))

    return res_A, max_eigvalue


def get_X(R):
    N = len(R)
    index = np.arange(0, N, 2)
    X = R ** 2
    X[index] = R[index]
    return X


def get_P(U, X, beta):
    N = len(X)
    res_A = np.matmul(U, X.transpose())
    res_B = LA.inv(np.matmul(X, X.transpose()) + beta * np.eye(N))
    P = np.matmul(res_A, res_B)
    shape = P.shape
    P1 = np.zeros(shape)
    P2 = np.zeros(shape)
    P1[:, np.arange(0, N, 2)] = P[:, np.arange(0, N, 2)]
    P2[:, np.arange(1, N + 1, 2)] = P[:, np.arange(1, N + 1, 2)]
    return P, P1, P2


def get_R(A, W_in, U, T):
    N = len(A)
    R = []
    r = np.zeros(N)
    R.append(r)
    for i in range(T - 1):
        r = r.reshape([-1, 1])
        u = U[:, i].reshape([-1, 1])
        r = np.tanh(np.matmul(A, r) + np.matmul(W_in, u))
        r = r.flatten()
        R.append(r)
    R = np.array(R).transpose()
    return R


def Predict_V(r, u, Length, P1, P2, A, W_in):
    res_V = []
    res_V.append(u.flatten())  # 预测的起点
    for i in range(Length - 1):
        r = np.tanh(np.matmul(A, r) + np.matmul(W_in, u))
        v = np.matmul(P1, r) + np.matmul(P2, r ** 2)
        res_V.append(v.flatten())
        
        u = v
    
    res_V = np.array(res_V).transpose()
    return res_V


def reinitial_r(U, length, A, W_in, D_r):
    r = np.zeros([D_r, 1])
    for i in range(length):
        u = U[:, i].reshape([-1, 1])
        r = np.tanh(np.matmul(A, r) + np.matmul(W_in, u))
    return r


def Predict_V_by_reinitial(r, u, Length, P1, P2, A, W_in, tau, D_r):
    res_V = []
    start = 0
    end = tau
    while True:
        now_V = Predict_V(r, u, end - start, P1, P2, A, W_in)
        r = reinitial_r(now_V, end - start, A, W_in, D_r)
        u = np.matmul(P1, r) + np.matmul(P2, r ** 2)
        
        if start == 0:
            res_V = now_V
        else:
            res_V = np.hstack((res_V, now_V))
        
        start += tau
        end += tau
        if start >= Length:
            break
        if end > Length:
            end = Length
        print('(start=%d, end=%d)' % (start, end))
    
    return res_V


def plot_result(t, U, test_t, True_V, V, Pred_V, fig_name=[1, 2, 3]):
    name = ['x(t)', 'y(t)', 'z(t)']
    for i in range(3):
        plt.figure(fig_name[i])
        plt.subplot(2, 2, 1)
        plt.plot(t, U[i, :])
        plt.plot(test_t, True_V[i, :], 'k-')
        plt.title('True date')
        plt.xlabel('time t')
        plt.ylabel(name[i], fontsize=24)
        
        # print(Pred_V.shape) #, Pred_V[i, :].shape)
        
        plt.subplot(2, 2, 2)
        plt.plot(t, V[i, :], 'r-')
        plt.plot(test_t, Pred_V[i, :], 'g-')
        plt.title('RC fitting')
        plt.xlabel('time t')
        plt.ylabel(name[i], fontsize=24)
        
        plt.subplot(2, 1, 2)
        plt.plot(t, U[i, :], 'b--')
        plt.plot(t, V[i, :], 'r-')
        plt.plot(test_t, True_V[i, :], 'k-')
        plt.plot(test_t, Pred_V[i, :], 'g-')
        
        plt.xlabel('time t')
        plt.ylabel(name[i], fontsize=24)
        plt.legend(['True data', 'RC fitting', 'pred. True data', 'RC Pred. data'])
        
        plt.title(name[i], fontsize=24)


def general_future_data_and_prediction(t, Length, V, R, P1, P2, A, W_in):
    delt_t = t[1] - t[0]
    test_t = np.array([t[-1] + delt_t * (i + 1) for i in range(Length)])
    
    u = V[:, -1].reshape([-1, 1])
    r = R[:, -1].reshape([-1, 1])
    True_V = get_lorenz_data(u, test_t)
    Pred_V = Predict_V(r, u, Length, P1, P2, A, W_in)
    return True_V, Pred_V

def plot_True_Pred(True_V, Pred_V,  T, Length):
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.subplot(3, 1, 1)
    plt.imshow(True_V, cmap=plt.jet())
    plt.title('Traing Length=%d, Following prediction steps=%d'%(T,Length), fontsize=20)
    
    plt.subplot(3, 1, 2)
    plt.imshow(Pred_V, cmap=plt.jet())
    
    plt.subplot(3, 1, 3)
    plt.imshow(Pred_V-True_V, cmap=plt.jet(), vmin=-2,vmax=2)

    cax = plt.axes([0.85, 0.3, 0.015, 0.4])
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    # plt.colorbar()
    
def get_next_True_Pred(Length, t, U, V, R, P1, P2, A, W_in):
    
    delt_t = t[1] - t[0]
    test_t = np.array([t[-1] + delt_t * (i + 1) for i in range(Length)])
    
    u = V[:, -1].reshape([-1, 1])
    r = R[:, -1].reshape([-1, 1])
    True_V = get_lorenz_data(u, test_t)
    Pred_V = Predict_V(r, u, Length, P1, P2, A, W_in)
    return True_V, Pred_V


def get_next_Pred(Length, V, R, P1, P2, A, W_in):
    u = V[:, -1].reshape([-1, 1])
    r = R[:, -1].reshape([-1, 1])
    Pred_V = Predict_V(r, u, Length, P1, P2, A, W_in)
    return Pred_V

def plot_for_creat_model(t, U, V, R, P1, P2, A, W_in):
    fontsize = 20
    Length = 100
    
    delt_t = t[1] - t[0]
    test_t = np.array([t[-1] + delt_t * (i + 1) for i in range(Length)])
    
    u = V[:, -1].reshape([-1, 1])
    r = R[:, -1].reshape([-1, 1])
    True_V = get_lorenz_data(u, test_t)
    Pred_V = Predict_V(r, u, Length, P1, P2, A, W_in)
    
    name = ['x(t)', 'y(t)', 'z(t)']
    plt.subplots_adjust(hspace=0.3, left=0.25)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        # plt.subplot(2, 2, 1)
        # plt.plot(t, U[i, :])
        # plt.plot(test_t, True_V[i, :], 'k-')
        # plt.title('True date')
        # plt.xlabel('time t')
        # plt.ylabel(name[i], fontsize=24)
        #
        # # print(Pred_V.shape) #, Pred_V[i, :].shape)
        #
        # plt.subplot(2, 2, 2)
        # plt.plot(t, V[i, :], 'r-')
        # plt.plot(test_t, Pred_V[i, :], 'g-')
        # plt.title('RC fitting')
        # plt.xlabel('time t')
        # plt.ylabel(name[i], fontsize=24)
        
        # plt.subplot(2, 1, 2)
        plt.plot(t, U[i, :], 'b--')
        plt.plot(t, V[i, :], 'r-')
        plt.plot(test_t, True_V[i, :], 'k-')
        plt.plot(test_t, Pred_V[i, :], 'g-')
        
        plt.xlabel('time t', fontsize=fontsize)
        plt.ylabel(name[i], fontsize=fontsize)
        # plt.legend(['True data', 'RC fitting', 'pred. True data', 'RC Pred. data'])
        
        if i == 0:
            plt.legend(['True data', 'RC fitting', 'pred. True data', 'RC Pred. data'],
                       loc='center', ncol=1, shadow=True, fontsize=fontsize,
                       bbox_to_anchor=(-0.2, -0.8))
            plt.title('Data length=' + str(len(t)) + '(for RC fitting), '
                      + 'Prediction length=' + str(Length),
                      fontsize=fontsize)


def get_mean_error(A, B):
    C = np.abs(A - B)
    fenzi = np.cumsum(C, axis=1)
    fenmu = np.tile(np.arange(1, C.shape[1] + 1), (C.shape[0], 1))
    return fenzi / fenmu


def plot_result_error(A, TT, N_experiment, D_r):
    fontsize = 20
    Num_T, D_in, step = A.shape
    # print('Num_T, D_in, step: ',Num_T, D_in, step)
    name = ['Direction x', 'Direction y', 'Direction z']
    shape = ['ro-', 'bs-', 'gd-', 'mv-', 'c<-', 'k+-', 'y*-', 'p-', 'h-']
    for i in range(D_in):
        plt.figure(i + 1)
        leg = []
        for j in range(Num_T):
            T = TT[j]
            # print('lalala shape: ',np.arange(1,step+1).shape,A[j,i,:].shape)
            plt.plot(np.arange(1, step + 1), A[j, i, :], shape[j])
            leg.append('Training Data length ' + str(T))
        plt.legend(leg, fontsize=fontsize - 6, shadow=True, loc='upper left')
        plt.xlabel('Prediction steps', fontsize=fontsize)
        plt.ylabel('Prediction Mean absolute error', fontsize=fontsize)
        # plt.title(str(step)+' prediction steps for '+ name[i]+
        #           ' (for %d experiments)'%N_experiment,
        #           fontsize=fontsize)
        plt.title(str(step) + ' prediction steps for ' + name[i] +
                  ' (for %d experiments) with %d hidden neurons'
                  % (N_experiment, D_r),
                  fontsize=fontsize)
