import math
import timeit

import geotorch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)

colors = [
    [107 / 256, 161 / 256, 255 / 256],
    [255 / 255, 165 / 255, 0],
    [233 / 256, 110 / 256, 236 / 256],
    [0.6, 0.4, 0.8],
    [0.0, 0.0, 1.0],
    [0.55, 0.71, 0.0],
    [0.99, 0.76, 0.8],
    [0.93, 0.53, 0.18],
    [11 / 255, 132 / 255, 147 / 255],
    [204 / 255, 119 / 255, 34 / 255],
]


class kepler(nn.Module):
    dim = 4

    def forward(self, t, X):
        dx = torch.zeros_like(X)
        x, y, a, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        dx[:, 0] = a
        dx[:, 1] = b
        dist = (x ** 2 + y ** 2) ** 1.5
        dx[:, 2] = -x / dist
        dx[:, 3] = -y / dist
        return dx


def t_energy(X):
    x, y, a, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (a ** 2 + b ** 2) / 2 - 1 / np.sqrt(x ** 2 + y ** 2)


def k_energy(X):
    x, y, a, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (a ** 2 + b ** 2) / 2


def p_energy(X):
    x, y, a, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return -1 / np.sqrt(x ** 2 + y ** 2)


def angular(X):
    x, y, a, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return x * b - y * a


def coord_MSE(X, Y):
    return np.sum((X - Y)[:, :2] ** 2, axis=1)


def generate_kepler_data(save_path="./data/true_y_12_300.npy", n=300, t_end=12):
    y0 = torch.tensor([[1.0, 0.0, 0.0, 0.9]])
    t = torch.linspace(0, t_end, n)
    y = odeint(kepler(), y0, t, atol=1e-8, rtol=1e-8).detach().numpy()[:, 0, :]
    np.save(save_path, y)
    return y


def generate_plot(K, X, T, n):
    Y = np.zeros([n, len(K)])
    X = X.detach().numpy()
    T = T.detach().numpy()
    Y[0, :] = T[0, :]

    for i in range(n - 1):
        x = Y[i, :].reshape(-1, 1)
        Y[i + 1, :] = np.matmul(K, x).T[0]

    plot_8(T, Y, X)


def plot_8(X=None, Y=None, Z=None, W=None):
    fontsize = 15
    labelsize = 13

    if X is None:
        X = np.load("./data/true_0.03.npy")
    if Y is None:
        Y = np.load("./data/noko_0.03.npy")
    if Z is None:
        Z = np.load("./data/hnn_0.03.npy")
    if W is None:
        W = np.load("./data/edmd_0.03.npy")

    plt.subplot(241)
    plt.plot(X[:, 0], X[:, 1], label="True", color="black")
    plt.plot(Y[:, 0], Y[:, 1], label="HNKO", color=colors[2])
    plt.ylabel("Trajectory", fontsize=fontsize)
    plt.title("HNKO", fontsize=fontsize)
    plt.xlabel(r"$x$", fontsize=fontsize)
    plt.xticks([-1, 0, 1], fontsize=labelsize)
    plt.yticks([-1, 0, 1], fontsize=labelsize)
    plt.legend(frameon=False)

    plt.subplot(242)
    plt.plot(X[:, 0], X[:, 1], label="True", color="black")
    plt.plot(Z[:, 0], Z[:, 1], label="HNN", color=colors[1])
    plt.title("HNN", fontsize=fontsize)
    plt.xticks([-1, 0, 1], fontsize=labelsize)
    plt.yticks([-1, 0, 1], fontsize=labelsize)
    plt.legend(frameon=False)

    plt.subplot(243)
    plt.plot(X[:, 0], X[:, 1], label="True", color="black")
    plt.plot(W[:, 0], W[:, 1], label="EDMD", color=colors[0])
    plt.title("EDMD", fontsize=fontsize)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.legend(frameon=False)

    plt.subplot(244)
    plt.plot(np.arange(len(coord_MSE(X, Y))), coord_MSE(X, Y), color=colors[2], label="HNKO")
    plt.plot(np.arange(len(coord_MSE(X, Z))), coord_MSE(X, Z), color=colors[1], label="HNN")
    plt.plot(np.arange(len(coord_MSE(X, W))), coord_MSE(X, W), color=colors[0], label="EDMD")
    plt.xticks([0, 25, 50], [0, 2.5, 5], fontsize=labelsize)
    plt.yticks([0, 0.05, 0.1], fontsize=labelsize)
    plt.title("MSE(Coordinates)", fontsize=fontsize)
    plt.legend(frameon=False)

    plt.subplot(245)
    for data, color in [(Y, colors[2]), (X, "black")]:
        plt.plot(np.arange(len(k_energy(data))), k_energy(data), label="Kinetic" if data is Y else None, color=color, ls="--")
        plt.plot(np.arange(len(p_energy(data))), p_energy(data), label="Potential" if data is Y else None, color=color, ls="dotted")
        plt.plot(np.arange(len(t_energy(data))), t_energy(data), label="Total" if data is Y else None, color=color, ls="-")
    plt.ylabel("Energy", fontsize=fontsize)
    plt.xlabel("Time", fontsize=fontsize)
    plt.yticks([-1.5, 0.0, 1.5], fontsize=labelsize)
    plt.xticks([0, 25, 50], [0, 2.5, 5], fontsize=labelsize)
    plt.legend(frameon=False, loc=5)

    plt.subplot(246)
    for data, color in [(Z, colors[1]), (X, "black")]:
        plt.plot(np.arange(len(k_energy(data))), k_energy(data), label="Kinetic" if data is Z else None, color=color, ls="--")
        plt.plot(np.arange(len(p_energy(data))), p_energy(data), label="Potential" if data is Z else None, color=color, ls="dotted")
        plt.plot(np.arange(len(t_energy(data))), t_energy(data), label="Total" if data is Z else None, color=color, ls="-")
    plt.yticks([-1.5, 0.0, 1.5], fontsize=labelsize)
    plt.xticks([0, 25, 50], [0, 2.5, 5], fontsize=labelsize)
    plt.legend(frameon=False, loc=5)

    plt.subplot(247)
    for data, color in [(W, colors[0]), (X, "black")]:
        plt.plot(np.arange(len(k_energy(data))), k_energy(data), label="Kinetic" if data is W else None, color=color, ls="--")
        plt.plot(np.arange(len(p_energy(data))), p_energy(data), label="Potential" if data is W else None, color=color, ls="dotted")
        plt.plot(np.arange(len(t_energy(data))), t_energy(data), label="Total" if data is W else None, color=color, ls="-")
    plt.yticks([-1.5, 0.0, 1.5], fontsize=labelsize)
    plt.xticks([0, 25, 50], [0, 2.5, 5], fontsize=labelsize)
    plt.legend(frameon=False, loc=5)

    plt.subplot(248)
    plt.plot(np.arange(len(t_energy(X))), np.sqrt((t_energy(X) - t_energy(Y)) ** 2), color=colors[2], label="HNKO")
    plt.plot(np.arange(len(t_energy(Y))), np.sqrt((t_energy(X) - t_energy(Z)) ** 2), color=colors[1], label="HNN")
    plt.plot(np.arange(len(t_energy(Z))), np.sqrt((t_energy(X) - t_energy(W)) ** 2), color=colors[0], label="EDMD")
    plt.xticks([0, 25, 50], [0, 2.5, 5], fontsize=labelsize)
    plt.yticks([0, 0.1, 0.2], fontsize=labelsize)
    plt.title("MSE(Total Energy)", fontsize=fontsize)
    plt.legend(frameon=False)
    plt.show()


class threebody(nn.Module):
    dim = 12

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        x1, y1, x2, y2, x3, y3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        a1, b1, a2, b2, a3, b3 = x[:, 6], x[:, 7], x[:, 8], x[:, 9], x[:, 10], x[:, 11]

        dx[:, 0] = a1
        dx[:, 1] = b1
        dx[:, 2] = a2
        dx[:, 3] = b2
        dx[:, 4] = a3
        dx[:, 5] = b3

        r12 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5
        r13 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 1.5
        r23 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 1.5

        dx[:, 6] = -(x1 - x2) / r12 - (x1 - x3) / r13
        dx[:, 7] = -(y1 - y2) / r12 - (y1 - y3) / r13
        dx[:, 8] = -(x2 - x1) / r12 - (x2 - x3) / r23
        dx[:, 9] = -(y2 - y1) / r12 - (y2 - y3) / r23
        dx[:, 10] = -(x3 - x1) / r13 - (x3 - x2) / r23
        dx[:, 11] = -(y3 - y1) / r13 - (y3 - y2) / r23
        return dx


def hami(x):
    x1, y1, x2, y2, x3, y3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    a1, b1, a2, b2, a3, b3 = x[:, 6], x[:, 7], x[:, 8], x[:, 9], x[:, 10], x[:, 11]

    kinetic = (a1 ** 2 + b1 ** 2 + a2 ** 2 + b2 ** 2 + a3 ** 2 + b3 ** 2) / 2
    potential = (
        -1 / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        -1 / np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        -1 / np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    )
    return kinetic + potential


def state_error(true, pred):
    return np.sqrt(np.sum((true - pred) ** 2, axis=1))


def rotate2d(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ p.reshape(2, 1)).squeeze()


def random_config(nu=0.02, min_radius=0.9, max_radius=1.2):
    p1 = np.array([0.5, 0.5])
    r = min_radius
    p1 *= r / np.sqrt(np.sum(p1 ** 2))
    p2 = rotate2d(p1, theta=2 * np.pi / 3)
    p3 = rotate2d(p2, theta=2 * np.pi / 3)

    v1 = rotate2d(p1, theta=np.pi / 2)
    v1 = v1 / r ** 1.5
    v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))
    v2 = rotate2d(v1, theta=2 * np.pi / 3)
    v3 = rotate2d(v2, theta=2 * np.pi / 3)

    v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
    v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
    v3 *= 1 + nu * (2 * np.random.rand(2) - 1)
    p1 += nu * (2 * np.random.rand(2))
    p2 += nu * (2 * np.random.rand(2))
    p3 += nu * (2 * np.random.rand(2))

    return np.concatenate((p1, p2, p3, v1, v2, v3))


def fix_config(nu=0.02, min_radius=0.9, max_radius=1.2):
    p1 = np.array([0.5, 0.5])
    r = min_radius
    p1 *= r / np.sqrt(np.sum(p1 ** 2))
    p2 = rotate2d(p1, theta=2 * np.pi / 3)
    p3 = rotate2d(p2, theta=2 * np.pi / 3)

    v1 = rotate2d(p1, theta=np.pi / 2)
    v1 = v1 / r ** 1.5
    v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))
    v2 = rotate2d(v1, theta=2 * np.pi / 3)
    v3 = rotate2d(v2, theta=2 * np.pi / 3)

    return np.concatenate((p1, p2, p3, v1, v2, v3))


def sphere_coord(theta, phi):
    r = 1.0
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


def rotate3d(theta, phi):
    v1 = sphere_coord(theta, phi)
    v2 = sphere_coord(np.pi / 2, np.pi * 3 / 4)
    u = np.array([1.0, 0.0, 0.0])
    v3 = u - np.dot(u, v1) * v1 - np.dot(u, v2) * v2
    v3 = v3 / np.linalg.norm(v3, 2)
    return np.matrix([v3, v2, v1]).T


def transform(A, trajectory, t):
    trans_data = np.zeros([len(trajectory), 6])
    t = np.asarray(t).reshape(-1, 1)

    for i in range(3):
        time_aug = np.concatenate((trajectory[:, 2 * i:2 * i + 2], t), axis=1)
        trans = np.matmul(A, time_aug.T).T
        trans_data[:, 2 * i:2 * i + 2] = trans[:, :2]

    timeline = np.concatenate((np.zeros([len(trajectory), 2]), t), axis=1)
    timeline = np.matmul(A, timeline.T).T
    return trans_data, timeline[:, :2]


def subplot(data, timeline, leg=False):
    co1, co2, co3 = colors[0], colors[1], colors[2]
    width = 1
    style = "solid"

    plt.plot(timeline[:, 0], timeline[:, 1], c="grey", label="Time")
    plt.plot(data[:, 0], data[:, 1], color=co1, ls=style, lw=width, label=r"$\mathbf{q}_1$")
    plt.plot(data[:, 2], data[:, 3], color=co2, ls=style, lw=width, label=r"$\mathbf{q}_2$")
    plt.plot(data[:, 4], data[:, 5], color=co3, ls=style, lw=width, label=r"$\mathbf{q}_3$")

    plt.scatter(data[-1, 0], data[-1, 1], color=co1, marker="o", s=100)
    plt.scatter(data[-1, 2], data[-1, 3], color=co2, marker="o", s=100)
    plt.scatter(data[-1, 4], data[-1, 5], color=co3, marker="o", s=100)

    plt.xticks([])
    plt.yticks([])

    if leg:
        plt.legend(frameon=False, fontsize=10)


def subsubplot(data):
    left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
    plt.axes([bottom, left, width, height])
    plt.plot(data[:, 0], data[:, 1], color=colors[0])


def plot0():
    n = 300
    y0 = torch.from_numpy(fix_config()).view([1, 12])
    t = torch.linspace(0, 30, n)
    true_y = odeint(threebody(), y0, t, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()

    A = rotate3d(-np.pi / 50, np.pi / 4)
    new_y, timeline = transform(A, true_y, t.detach().numpy())

    plt.plot(new_y[:, 0], new_y[:, 1], c="blue", ls="dashed", lw=3)
    plt.plot(new_y[:, 2], new_y[:, 3], c="navy", ls="dashed", lw=3)
    plt.plot(new_y[:, 4], new_y[:, 5], c="royalblue", ls="dashed", lw=3)
    plt.plot(timeline[:, 0], timeline[:, 1], c="black")
    plt.scatter(new_y[-1, 0], new_y[-1, 1], c="blue", marker="o", s=100)
    plt.show()


def generate_threebody_data(
    train_path="./data/train_y_5_100.npy",
    true_path="./data/true_y_90_1800.npy",
):
    y0 = torch.from_numpy(fix_config()).view([1, 12])

    t_train = torch.linspace(0, 5, 100)
    train_y = odeint(threebody(), y0, t_train, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()

    t_true = torch.linspace(0, 90, 1800)
    true_y = odeint(threebody(), y0, t_true, atol=1e-8, rtol=1e-8)[:, 0, :].detach().numpy()

    np.save(train_path, train_y)
    np.save(true_path, true_y)
    return train_y, true_y


def plot_threebody_comparison():
    mpl.rcParams["font.sans-serif"] = "NSimSun,Times New Roman"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"

    fontsize = 15
    labelsize = 15

    true = np.load("./data/true_y_90_1800.npy")[:1000]
    hnn = np.load("./data/hnn_90_1800_0.03.npy")[:1000]
    hnko = np.load("./data/HNKO_90_1800_0.03.npy")[:1000]
    edmd = np.load("./data/enc_edmd_90_1800_0.03.npy")[:1000]

    t = np.linspace(0, 50, 1000)
    A = rotate3d(-np.pi / 50, np.pi * 0.3)

    t_true, timeline = transform(A, true, t)
    t_hnn, _ = transform(A, hnn, t)
    t_hnko, _ = transform(A, hnko, t)
    t_edmd, _ = transform(A, edmd, t)

    plt.subplot(321)
    subplot(t_true, timeline)
    plt.title("Truth", fontsize=fontsize)

    plt.subplot(322)
    subplot(t_hnn, timeline)
    plt.title("HNN", fontsize=fontsize)

    plt.subplot(323)
    subplot(t_edmd, timeline)
    plt.title("EDMD", fontsize=fontsize)

    plt.subplot(324)
    subplot(t_hnko, timeline)
    plt.title("HNKO", fontsize=fontsize)

    plt.subplot(325)
    plt.plot(t, state_error(true, hnn), label="HNN")
    plt.plot(t, state_error(true, edmd), label="EDMD")
    plt.plot(t, state_error(true, hnko), label="HNKO")
    plt.xticks([0, 25, 50], fontsize=labelsize)
    plt.yticks([0, 8, 16], fontsize=labelsize)
    plt.xlabel("Time", fontsize=fontsize)
    plt.ylabel("State error", fontsize=fontsize)

    plt.subplot(326)
    plt.plot(t, hami(hnn), label="HNN")
    plt.plot(t, hami(edmd), label="EDMD")
    plt.plot(t, hami(hnko), label="HNKO")
    plt.plot(t, hami(true), color="black", label="Truth", ls=(0, (3, 3)))
    plt.xticks([0, 25, 50], fontsize=labelsize)
    plt.yticks([-2, 0, 5], fontsize=labelsize)
    plt.legend(ncol=4, bbox_to_anchor=[0, -0.3], fontsize=10, frameon=False)
    plt.xlabel("Time", fontsize=fontsize)
    plt.ylabel("Energy", fontsize=fontsize)
    plt.show()


def plot_threebody_error():
    mpl.rcParams["font.sans-serif"] = "NSimSun,Times New Roman"
    plt.rcParams["ytick.direction"] = "in"

    labelsize = 15
    true = np.load("./data/true_y_90_1800.npy")[:1000]
    hnko = np.load("./data/HNKO_90_1800_0.03.npy")[:1000]
    edmd = np.load("./data/enc_edmd_90_1800_0.03.npy")[:1000]
    t = np.linspace(0, 50, 1000)

    plt.plot(t, state_error(true, edmd), label="EDMD")
    plt.plot(t, state_error(true, hnko), label="HNKO")
    plt.xticks([0, 25, 50], fontsize=labelsize)
    plt.yticks([0, 0.5], fontsize=labelsize)
    plt.show()


generate = generate_kepler_data
plot = plot_threebody_comparison
