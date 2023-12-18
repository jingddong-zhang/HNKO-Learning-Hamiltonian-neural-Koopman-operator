import argparse
import os
import random
import timeit

import dgl
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from functions import *
from torchgde import GCNLayer, GDEFunc, ODEBlock, accuracy
from torchgde.models.odeblock import ODESolvers
torch.set_default_dtype(torch.float64)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument(
        "--dataset",
        default="cora",
        const="cora",
        nargs="?",
        choices=("cora", "citeseer", "pubmed"),
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dropout", type=float, default=0.9)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=-1)
    # parser.add_argument("--guide", action=argparse.BooleanOptionalAction)
    # parser.add_argument("--fast", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjoint",default=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--solver",
        default="rk4",
        const="rk4",
        nargs="?",
        choices=tuple(e.value for e in ODESolvers),
    )
    args = parser.parse_args()
    return args





graph = dgl.graph(([0,0,0,1,1,1,2,2,2], [0,1,2,0,1,2,0,1,2]))
graph.edata['x'] = torch.ones(graph.num_edges())
degrees = graph.in_degrees().float()
norm = torch.pow(degrees, -0.5)
norm[torch.isinf(norm)] = 0
graph.ndata["norm"] = norm.unsqueeze(1)


hidden_channels = 32
gnn = nn.Sequential(
    GCNLayer(
        g=graph,
        in_feats=4,
        out_feats=hidden_channels,
        activation=nn.Softplus(),
        dropout=0.9,
    ),
    GCNLayer(
        g=graph,
        in_feats=hidden_channels,
        out_feats=4,
        activation=nn.Softplus(),
        dropout=0.9,
    ),
)
class GNN(nn.Module):
    def __init__(self,hidden_channels,layers):
        super().__init__()
        self.layers = layers
        self.layer1 =  GCNLayer(g=graph,
                                in_feats=4,
                                out_feats=hidden_channels,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.layer2 =  GCNLayer(g=graph,
                                in_feats=hidden_channels,
                                out_feats=hidden_channels,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.layer3 =  GCNLayer(g=graph,
                                in_feats=hidden_channels,
                                out_feats=hidden_channels,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.layer4 =  GCNLayer(g=graph,
                                in_feats=hidden_channels,
                                out_feats=hidden_channels,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.layer5 =  GCNLayer(g=graph,
                                in_feats=hidden_channels,
                                out_feats=hidden_channels,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.layer6 =  GCNLayer(g=graph,
                                in_feats=hidden_channels,
                                out_feats=1,
                                activation=nn.Softplus(),
                                dropout=0.9,
                            )
        self.linear = nn.Linear(3,1)

    def forward(self, x):
        h1 = self.layer1(x)
        if self.layers == 2:
            h2 = self.layer2(h1).transpose(-2,-1)
            h3 = self.linear(h2)
            return h3
        if self.layers == 4:
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            h4 = self.layer4(h3).transpose(-2,-1)
            h5 = self.linear(h4)
            return h5
        if self.layers == 6:
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            h4 = self.layer4(h3)
            h5 = self.layer5(h4)
            h6 = self.layer6(h5).transpose(-2, -1)
            h7 = self.linear(h6)
            return h3


gnn = GNN(hidden_channels,2)
gdefunc = GDEFunc(gnn)
gde = ODEBlock(
    odefunc=gdefunc,
    method='rk4',
    atol=1e-3,
    rtol=1e-4,
    use_adjoint=True,
)

y0 = torch.from_numpy(fix_config()).view([1, 12])
t = torch.linspace(0, 5, 100)
true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :]
true_y = true_y.requires_grad_(True)
pos, moment = true_y[:, :6], true_y[:, 6:]
feat1, feat2, feat3 = torch.cat((pos[:, 0:2], moment[:, 0:2]), dim=1), torch.cat((pos[:, 2:4], moment[:, 2:4]),
                                                                                 dim=1), torch.cat((pos[:, 4:6], moment[:, 4:6]), dim=1)
x0 = torch.cat((feat1[0:1], feat2[0:1], feat3[0:1]), dim=0)

# print(x0.shape)
# x0 = torch.randn([2,2])
# t = torch.linspace(0,1,20)
# true_y = gdefunc(0,x)

# pred_y = gde.cforward(x0, t)
# yy = torch.zeros([100,12])
# yy[:,0:2],yy[:,6:8] = pred_y[:,0,0:2],pred_y[:,0,2:4]
# yy[:,2:4],yy[:,8:10] = pred_y[:,1,0:2],pred_y[:,1,2:4]
# yy[:,4:6],yy[:,10:12] = pred_y[:,2,0:2],pred_y[:,2,2:4]

# true_y,pred_y=true_y.detach().numpy(),yy.detach().numpy()
# plt.plot(true_y[:,0],true_y[:,1])
# plt.plot(pred_y[:,0],pred_y[:,1])
# plt.show()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ode = gde

    def forward(self, x: torch.Tensor, t):
        x = self.ode.cforward(x, t)
        return x

def numer_deri(t,X):
    dt = t[1]-t[0]
    X_r = torch.cat((X[1:],X[:1]),dim=0)
    X_l = torch.cat((X[-1:],X[:-1]),dim=0)
    dX = (X_r-X_l)/(dt*2)
    return dX

def ana_deri(t,X):
    func = threebody()
    dX = func(0.0,X)
    return dX

def get_batch(true_y):
    s = torch.from_numpy(np.random.choice(np.arange(100-10, dtype=np.int64), 10, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    yy = torch.cat((batch_y0[:,:6].view(10,3,2),batch_y0[:,6:].view(10,3,2)),dim=2)

    batch_t = t[:10]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(10)], dim=0)  # (T, M, D)
    return yy, batch_t, batch_y



def main(hidden_channels,layers,lr):
    torch.manual_seed(369)
    np.random.seed(369)
    y0 = torch.from_numpy(fix_config()).view([1, 12]).to(device)
    t = torch.linspace(0, 5, 100).to(device)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].to(device)
    true_y = true_y.requires_grad_(True).to(device)
    # dy = numer_deri(t, true_y)[1:-1].to(device)
    dy = ana_deri(t,true_y).to(device)
    dq, dp = dy[:, :6], dy[:, 6:]
    train_y = true_y + torch.from_numpy(np.random.normal(0,0.03,true_y.shape))
    feat = torch.cat((train_y[:,:6].view(100,3,2),train_y[:,6:].view(100,3,2)),dim=2)
    pos, moment = true_y[:, :6], true_y[:, 6:]
    feat1, feat2, feat3 = torch.cat((pos[:,0:2],moment[:,0:2]),dim=1),torch.cat((pos[:,2:4],moment[:,2:4]),dim=1),torch.cat((pos[:,4:6],moment[:,4:6]),dim=1)
    x0 = torch.cat((feat1[0:1], feat2[0:1],feat3[0:1]), dim=0).to(device)
    print(x0.shape,x0-feat[0])


    # device = torch.device("cpu")
    gnn = GNN(hidden_channels,layers)
    model = GDEFunc(gnn).to(device)
    # model = Model().to(device)
    # lr = 5e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Loss = []
    start = timeit.default_timer()
    for i in tqdm(range(1000), leave=False):
        model.train()
        optimizer.zero_grad()

        # batch_y0, batch_t, batch_y = get_batch(true_y)
        # loss = 0
        # for j in range(len(batch_y0)):
        #     outputs = model(batch_y0[j], batch_t)
        #     yy = torch.zeros([outputs.shape[0], 12])
        #     yy[:, 0:2], yy[:, 6:8] = outputs[:, 0, 0:2], outputs[:, 0, 2:4]
        #     yy[:, 2:4], yy[:, 8:10] = outputs[:, 1, 0:2], outputs[:, 1, 2:4]
        #     yy[:, 4:6], yy[:, 10:12] = outputs[:, 2, 0:2], outputs[:, 2, 2:4]
        #     pred_y = yy
        #     loss += torch.mean(torch.abs(pred_y[-1,:]-batch_y[-1,j,:]))
        loss = 0
        for j in range(len(true_y)):
            outputs = model.derivative(0,feat[j])
            H_q, H_p = outputs[:,:2].reshape(-1,6), outputs[:, 2:].reshape(-1,6)
            loss += torch.sum(torch.sqrt(torch.sum((H_p - dq[j:j+1]) ** 2, dim=1))) \
                   + torch.sum(torch.sqrt(torch.sum((H_q + dp[j:j+1]) ** 2, dim=1)))

        loss.backward(retain_graph=True)
        optimizer.step()
        Loss.append(loss)
        if loss == min(Loss):
            best_model = model.state_dict()

        print(f'({hidden_channels},{layers},{lr}), current loss {loss}')


    model.load_state_dict(best_model)
    gde = ODEBlock(
        odefunc=model.sympletic,
        method='rk4',
        atol=1e-3,
        rtol=1e-4,
        use_adjoint=False,
    )
    t = torch.linspace(0,90,1800)
    outputs = gde.cforward(x0, t)
    yy = torch.zeros([outputs.shape[0], 12])
    yy[:, 0:2], yy[:, 6:8] = outputs[:, 0, 0:2], outputs[:, 0, 2:4]
    yy[:, 2:4], yy[:, 8:10] = outputs[:, 1, 0:2], outputs[:, 1, 2:4]
    yy[:, 4:6], yy[:, 10:12] = outputs[:, 2, 0:2], outputs[:, 2, 2:4]
    true_y,pred_y=true_y.detach().numpy(),yy.detach().numpy()
    np.save('./data/GCN_data/HGCN_90_1800_0.03 layer={} lr={} hidden={}'.format(layers,lr,hidden_channels),pred_y)
    end = timeit.default_timer()
    print(f'Total time:{end-start}')

    # plt.subplot(121)
    # plt.plot(true_y[:,0],true_y[:,1])
    # plt.plot(true_y[:, 2], true_y[:, 3])
    # plt.plot(true_y[:, 4], true_y[:, 5])
    # plt.subplot(122)
    # plt.plot(pred_y[:,0],pred_y[:,1])
    # plt.plot(pred_y[:, 2], pred_y[:, 3])
    # plt.plot(pred_y[:, 4], pred_y[:, 5])
    # plt.show()


def main_different_data():
    '''
    bset model
    '''
    hidden_channels = 64
    layers = 4
    lr = 5e-3
    torch.manual_seed(369)
    np.random.seed(369)

    # for _, num in enumerate([500,1000,5000,10000]):
    for _, sigma in enumerate([0.01,0.05]):
        num = 100
        # sigma = 0.03
        y0 = torch.from_numpy(fix_config()).view([1, 12]).to(device)
        t = torch.linspace(0, 5, num).to(device)
        true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].to(device)
        true_y = true_y.requires_grad_(True).to(device)
        dy = ana_deri(t, true_y).to(device)
        dq, dp = dy[:, :6], dy[:, 6:]
        train_y = true_y + torch.from_numpy(np.random.normal(0, sigma, true_y.shape))
        feat = torch.cat((train_y[:, :6].view(num, 3, 2), train_y[:, 6:].view(num, 3, 2)), dim=2)
        pos, moment = true_y[:, :6], true_y[:, 6:]
        feat1, feat2, feat3 = torch.cat((pos[:, 0:2], moment[:, 0:2]), dim=1), torch.cat((pos[:, 2:4], moment[:, 2:4]),
                                                                                         dim=1), torch.cat(
            (pos[:, 4:6], moment[:, 4:6]), dim=1)
        x0 = torch.cat((feat1[0:1], feat2[0:1], feat3[0:1]), dim=0).to(device)
        print(x0.shape, x0 - feat[0])

        gnn = GNN(hidden_channels, layers)
        model = GDEFunc(gnn).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        Loss = []
        start = timeit.default_timer()
        for i in tqdm(range(1000), leave=False):
            model.train()
            optimizer.zero_grad()

            loss = 0
            s = np.random.choice(np.arange(num, dtype=np.int64), 100, replace=False)
            for j in s:
                outputs = model.derivative(0, feat[j])
                H_q, H_p = outputs[:, :2].reshape(-1, 6), outputs[:, 2:].reshape(-1, 6)
                loss += torch.sum(torch.sqrt(torch.sum((H_p - dq[j:j + 1]) ** 2, dim=1))) \
                        + torch.sum(torch.sqrt(torch.sum((H_q + dp[j:j + 1]) ** 2, dim=1)))

            loss.backward(retain_graph=True)
            optimizer.step()
            Loss.append(loss)
            if loss == min(Loss):
                best_model = model.state_dict()

            print(f'({hidden_channels},{layers},{lr}), current loss {loss}')

        model.load_state_dict(best_model)
        gde = ODEBlock(
            odefunc=model.sympletic,
            method='rk4',
            atol=1e-3,
            rtol=1e-4,
            use_adjoint=False,
        )
        t = torch.linspace(0, 90, 1800)
        outputs = gde.cforward(x0, t)
        yy = torch.zeros([outputs.shape[0], 12])
        yy[:, 0:2], yy[:, 6:8] = outputs[:, 0, 0:2], outputs[:, 0, 2:4]
        yy[:, 2:4], yy[:, 8:10] = outputs[:, 1, 0:2], outputs[:, 1, 2:4]
        yy[:, 4:6], yy[:, 10:12] = outputs[:, 2, 0:2], outputs[:, 2, 2:4]
        true_y, pred_y = true_y.detach().numpy(), yy.detach().numpy()
        np.save('./data/GCN_data/HGCN_best length={} sigma={}'.format(num,sigma),pred_y)
        end = timeit.default_timer()
        print(f'Total time:{end - start}')


def num_model(hidden,layers,lr):
    class GNN(nn.Module):
        def __init__(self, hidden_channels, layers):
            super().__init__()
            self.layers = layers
            self.layer1 = GCNLayer(g=graph,
                                   in_feats=4,
                                   out_feats=hidden_channels,
                                   activation=nn.Softplus(),
                                   dropout=0.9,
                                   )
            self.layer2 = GCNLayer(g=graph,
                                   in_feats=hidden_channels,
                                   out_feats=hidden_channels,
                                   activation=nn.Softplus(),
                                   dropout=0.9,
                                   )
            self.layer3 = GCNLayer(g=graph,
                                   in_feats=hidden_channels,
                                   out_feats=hidden_channels,
                                   activation=nn.Softplus(),
                                   dropout=0.9,
                                   )
            self.layer4 = GCNLayer(g=graph,
                                   in_feats=hidden_channels,
                                   out_feats=hidden_channels,
                                   activation=nn.Softplus(),
                                   dropout=0.9,
                                   )
            self.linear = nn.Linear(3, 1)

        def forward(self, x):
            h1 = self.layer1(x)
            if self.layers == 4:
                h2 = self.layer2(h1)
                h3 = self.layer3(h2)
                h4 = self.layer4(h3).transpose(-2, -1)
                h5 = self.linear(h4)
                return h5
    model = GNN(hidden_channels,layers)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)



def calculate():
    true = np.load('./data/true_y_90_1800.npy')[300:1200]
    H1_list = [32,64,128]
    lr_list = [0.05,0.005,0.0005]
    layer_list = [2,4,6]
    error_list = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                H1 = H1_list[k]
                lr = lr_list[j]
                layer = layer_list[i]
                pred = np.load('./data/GCN_data/HGCN_90_1800_0.03 layer={} lr={} hidden={}.npy'.format(layer,lr,H1))[300:1200]
                error = np.sqrt(np.sum((true - pred) ** 2, axis=1)).mean()
                print(layer,lr,H1,error)
                if not np.isnan(error):
                    error_list.append(error)
                else:
                    error_list.append(error)
    print(min(error_list))
    for i,_ in enumerate(error_list):
        if i%9 == 0:
            print('\multirow{3}{*}{}',end=' ')
            print('& ${}$ '.format(2*(int(i/9)+1)))
        print('&${:.2f}$'.format(_),end=' ')
        if (i+1)%9==0:
            print('\\\\')
            print('\n')



if __name__ == "__main__":
    args = parse_args()
    # for lr in [5e-2,5e-3,5e-4]:
    #     main(128,2,lr)

    # for lr in [5e-2,5e-3,5e-4]:
    #     for layers in [2,4,6]:
    #         for hidden in [32,64,128]:
    #             main(hidden,layers,lr)

    # calculate()
    # num_model(64, 4, 5e-4)  # best model
    main_different_data()