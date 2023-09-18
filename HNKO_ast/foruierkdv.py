# Reference Scipy Cookbook Solve KdV
# Import libraries
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
import matplotlib.pyplot as plt
# import imvideo as imv
import time
from scipy import integrate
import torch
import torch.nn as nn

def main():
    '''main numerical KdV solution function'''
    # Set the size of the domain, and create the discretized grid.
    L = 50  # length of periodic boundary
    N = 300  # number of spatial steps
    dx = L / (N - 1.0)  # spatial step size
    x = np.linspace(0, (1 - 1.0 / N) * L, N)  # initialize x spatial axis

    main_start = time.time()
    # Set the initial conditions.
    # Not exact for two solitons on a periodic domain, but close enough...
    u0 = kdv_soliton_solution(x - 0.33 * L, 0.75)  # + kdv_soliton_solution(x-0.65*L, 0.4)

    # Set the time sample grid.
    T = 600
    t = np.linspace(0, T, 501)

    # sol = solve_kdv(kdv_model, u0, t, L, 5000)
    sol = solve_kdv(kdv_torch(), u0, t, L, 5000)
    print(sol.shape)
    np.save('./data/samples_{}_501'.format(N),{'t':t,'x':x,'u_x':sol})
    # H = [integrate.trapz(sol[i, :]**2, x) for i in range(len(sol))]
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))
    # plt.plot(t,H)
    # plt.show()
    plot(sol, L, T)
    # animate(sol, L, T, 50)
    return


# The KdV model using Fast Fourier Transform
def kdv_model(u, t, L):
    '''The KdV model using FFT
    Input:
            u       (float)     wave amplitude
            t       (float)     simulation duration
            L       (float)     X length for periodic boundaries
    Output:
            dudt    (float)     left side of the time differential
    '''
    # Compute the space differentials of u
    ux = psdiff(u, period=L)  # 1st order differential
    uxxx = psdiff(u, period=L, order=3)  # 3rd order differential

    dudt = -6 * u * ux - uxxx  # - 0.01*u                  #KdV model; time differential

    return dudt


class kdv_torch(nn.Module):
    def forward(self, t, x):
        # x: [b_size, d_dim]
        L = 50
        # ux = torch.diff(x,dim=1)
        # uxx = torch.diff(ux,dim=1)
        # uxxx = torch.diff(uxx,dim=1)
        # dudt = -6 * x * ux - uxxx
        # return dudt
        u = x[0,:].numpy()
        ux = psdiff(u, period=L)  # 1st order differential
        uxxx = psdiff(u, period=L, order=3)  # 3rd order differential
        dudt = -6 * u * ux - uxxx  # - 0.01*u                  #KdV model; time differential
        dx = torch.zeros_like(x)
        dx[0,:] = torch.from_numpy(dudt)
        return dx


# The analytical soliton solution to the KdV
def kdv_soliton_solution(x, c):
    '''The exact soliton solution to the KdV
    Input:
            x   (float)     variable 1
            c   (float)     variable 2
    Output:
            u   (float)     wave amplitude
    '''
    u = 0.5 * c * np.cosh(0.5 * np.sqrt(c) * x)**(-2)
    # u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)-0.005*x

    return u


def solve_kdv(model, u0, t, L, steps):
    '''Solve the KdV using Scipy odeint
    Input:
        model               kdv model
        u0      (float/int) initial amplitude
        t       (int)       time range
        L       (int)       periodic range
        steps   (int)       maximum steps allowed
    Output:
        sol     (array)     periodic solutions'''
    # sol = odeint(model, u0, t, args=(L,), mxstep=steps)

    from torchdiffeq import odeint_adjoint as odeint
    u0 = torch.from_numpy(u0).view(1,-1)
    t = torch.from_numpy(t)
    with torch.no_grad():
        sol = odeint(model, u0, t, method='bosh3')[:,0,:].detach().numpy()
    return sol


def plot(sol, rangeX, rangeT):
    '''plot the KdV solution
    Input:
        sol
        rangeX      (float/int)
        rangeT      (float/int)
    Output:
        None
    '''
    plt.figure(figsize=(8, 8))
    plt.imshow(sol, extent=[0, rangeX, 0, rangeT])  # test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('Korteweg-de Vries on a Periodic Domain')
    plt.show()

    return

if __name__ == "__main__":
    main()