import argparse
import os
import random
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from functions import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
print(device)

parser = argparse.ArgumentParser('CNN LSTM')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--lr', type=int, default=3e-3)
parser.add_argument('--in_channels', default=12)
parser.add_argument('--out_channels', default=32)
parser.add_argument('--hidden_size', default=32)
parser.add_argument('--output_size', default=12)
parser.add_argument('--num_layers', default=4)
parser.add_argument('--pred_size', default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

class CNN_LSTM(nn.Module):
    def __init__(self, dim,hidden,layers):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=hidden, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x[:,-args.pred_size:,:]
        return x



def get_batch(true_y):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size-args.batch_time-args.pred_size, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (B, D)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_traget = torch.stack([true_y[s + args.batch_time + i] for i in range(args.pred_size)], dim=0)  # (T, M, D)
    return batch_y.permute(1,0,2),batch_traget.permute(1,0,2) #(B,T,D)

def main(hidden_channels,layers,lr):
    torch.manual_seed(369)
    np.random.seed(369)
    y0 = torch.from_numpy(fix_config()).view([1, 12]).to(device)
    t = torch.linspace(0, 5, 100).to(device)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].to(device)
    true_y = true_y.requires_grad_(True).to(device)

    train_y = true_y + torch.from_numpy(np.random.normal(0,0.03,true_y.shape))

    dim = train_y.shape[1]
    model = CNN_LSTM(dim,hidden_channels,layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    Loss = []
    start = timeit.default_timer()
    for i in tqdm(range(1000), leave=False):
        model.train()
        optimizer.zero_grad()

        batch_y,batch_target = get_batch(true_y)
        output = model(batch_y)
        # print(output.shape,batch_target.shape)
        lossunc = torch.nn.MSELoss()
        loss = lossunc(output, batch_target)

        loss.backward(retain_graph=True)
        optimizer.step()
        Loss.append(loss)
        if loss == min(Loss):
            best_model = model.state_dict()

        print(f'({hidden_channels},{layers},{lr}), current loss {loss}')

    model.load_state_dict(best_model)
    length = 1700
    pred_y = torch.zeros([length+args.batch_time,dim])
    pred_y[:args.batch_time] = true_y[-args.batch_time:,:]
    for j in tqdm(range(length)):
        curr_y = pred_y[j:j+args.batch_time].unsqueeze(0)
        pred_y[j+args.batch_time] = model(curr_y)[0,0:1,:]

    true_y, pred_y = true_y.detach().numpy(), pred_y[args.batch_time:].detach().numpy()
    np.save('./data/CNN_data/CNN_90_1800_0.03 layer={} lr={} hidden={}'.format(layers,lr,hidden_channels),pred_y)

    # plt.subplot(121)
    # plt.plot(true_y[:,0],true_y[:,1])
    #
    # plt.subplot(122)
    # plt.plot(pred_y[:,0],pred_y[:,1])
    # plt.show()


def main_different_data():
    '''
    bset model
    '''
    hidden_channels = 128
    layers = 2
    lr = 5e-4
    torch.manual_seed(369)
    np.random.seed(369)

    # for _, num in enumerate([100,500,1000,5000,10000]):
    for _, sigma in enumerate([0.01,0.05]):
        num = 100
        # sigma = 0.03
        y0 = torch.from_numpy(fix_config()).view([1, 12]).to(device)
        t = torch.linspace(0, 5, num).to(device)
        true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].to(device)
        true_y = true_y.requires_grad_(True).to(device)

        train_y = true_y + torch.from_numpy(np.random.normal(0,sigma,true_y.shape))

        dim = train_y.shape[1]
        model = CNN_LSTM(dim,hidden_channels,layers).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        Loss = []
        start = timeit.default_timer()
        for i in tqdm(range(1000), leave=False):
            model.train()
            optimizer.zero_grad()

            batch_y,batch_target = get_batch(true_y)
            output = model(batch_y)
            # print(output.shape,batch_target.shape)
            lossunc = torch.nn.MSELoss()
            loss = lossunc(output, batch_target)

            loss.backward(retain_graph=True)
            optimizer.step()
            Loss.append(loss)
            if loss == min(Loss):
                best_model = model.state_dict()

            print(f'({hidden_channels},{layers},{lr}), current loss {loss}')

        model.load_state_dict(best_model)
        length = 1700
        pred_y = torch.zeros([length+args.batch_time,dim])
        pred_y[:args.batch_time] = true_y[-args.batch_time:,:]
        for j in tqdm(range(length)):
            curr_y = pred_y[j:j+args.batch_time].unsqueeze(0)
            pred_y[j+args.batch_time] = model(curr_y)[0,0:1,:]

        true_y, pred_y = true_y.detach().numpy(), pred_y[args.batch_time:].detach().numpy()
        np.save('./data/CNN_data/CNN_best length={} sigma={}'.format(num,sigma),pred_y)


def num_model(hidden,layers,lr):
    model = CNN_LSTM(12,hidden,layers)
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
                pred = np.load('./data/CNN_data/CNN_90_1800_0.03 layer={} lr={} hidden={}.npy'.format(layer,lr,H1))[200:1100]
                error = np.sqrt(np.sum((true - pred) ** 2, axis=1)).mean()
                # print(layer,lr,H1,error)
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
if __name__ == '__main__':
    # for lr in [5e-2,5e-3,5e-4]:
    #     for layers in [2,4,6]:
    #         for hidden in [32,64,128]:
    #             main(hidden,layers,lr)
    # calculate()
    # num_model(128,2,5e-4)
    main_different_data()