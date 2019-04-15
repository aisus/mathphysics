import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from mpmath import mp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from functools import wraps
import argparse

mp.dps = 15

T = 100
C = 0.5
D = 0.001
l = 30
alpha = 0.3
a = mp.sqrt(alpha / C)
xi = mp.pi / T


def with_dps(dps):
    """
        Param. decorator.
        When applied to function, changes mp.dps inside it.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            old_dps = mp.dps
            mp.dps = dps

            res = f(*args, **kwds)

            mp.dps = old_dps

            return res

        return wrapper

    return decorator


def phi(t):
    return 0.1 * mp.sin(xi * t)**2


def u(x, t, num_steps=10):
    return v(x, t, num_steps) + phi(t)


def v(x, t, num_steps=10):
    res = 0

    for k in range(num_steps):
        n = 2*k + 1 # n = 1,3,5,...
        res += v_n(n, x, t)

    return res


def v_n(n, x, t):
    return C_n(n, t) * X_n(n, x)


phi_0 = phi(0)


@with_dps(20)
def C_n(n, t):
    if n % 2 == 0:
        return 0

    if t == 0:
        return -phi_0 * 4 / (mp.pi * n)

    g = -((n * mp.pi * a / l)**2 + D / C)

    # put e^(g*t) in integrals, otherwise they will grow
    # very (VERY) fast, and result will be Inf
    #
    # def f1(tau):
    #     return 0.1 * mp.sin(xi * tau)**2 * mp.exp(g * (t - tau))
    #
    # def f2(tau):
    #     return 0.2 * xi * mp.sin(xi * tau) * mp.cos(xi * tau) * \
    #         mp.exp(g * (t - tau))
    #
    # I1, err = mp.quad(f1, [0, t], error=True)
    # if err > 1e-14:
    #     print('ERR')
    # logging.debug(f'I1 error = {err}')
    #
    # I2, err = mp.quad(f2, [0, t], error=True)
    # if err > 1e-14:
    #     print('ERR')
    # logging.debug(f'I2 error = {err}')
    #
    # res = mp.exp(g * t) * C_n(n, 0) - 4 * (D * I1 / C + I2) / (mp.pi * n)

    # analytical solution

    I1_a = mp.exp(-g*t)*(-g**2 + g**2*mp.cos(2*xi*t) -2*xi*g*mp.sin(2*xi*t) + 4*xi**2*mp.exp(g*t) -4*xi**2) / (20*(g**3 + 4*xi**2*g))
    I2_a = xi*mp.exp(-g*t)*(-g*mp.sin(2*xi*t) - 2*xi*mp.cos(2*xi*t) + 2*xi*mp.exp(g*t))/(10*(g**2 + 4*xi**2))

    res_a = mp.exp(g * t) * (C_n(n, 0) - 4 * (D * I1_a / C + I2_a)) / (mp.pi * n)

    return res_a


sin_vec = np.vectorize(mp.sin)


def X_n(n, x):
    return sin_vec(mp.pi * n * x / l)

def tridiag_solve(A, b):
    n = len(b)
    solution = [mp.mpf(0) for _ in range(n)]

    for i in range(1, n):
        A[i][i] -= A[i][i - 1] / A[i - 1][i - 1] * A[i - 1][i]
        b[i] -= A[i][i - 1] / A[i - 1][i - 1] * b[i - 1]

    solution[-1] = b[-1] / A[-1][-1]

    for i in range(n - 2, -1, -1):
        solution[i] = (b[i] - A[i][i + 1] * solution[i + 1]) / A[i][i]

    return solution


def check(h, tau):
    err = tau - 2 * C * h ** 2 / (2 * alpha + D * h ** 2)
    return (err + 1e-6 <= 0), err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('-n', '--num', help='number of elements to sum up', type=int, default=150)
    parser.add_argument('-c', '--cycles', help='number of full cycles to simulate', type=int, default=1)
    parser.add_argument('-x', '--x', help='number of points in x', type=int, default=11)
    parser.add_argument('-t', '--t', help='number of points in t', type=int, default=21)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    num_steps = args.num
    x = np.linspace(0, l, args.x)
    t = np.linspace(0, args.cycles * T, args.cycles*args.t)

    n = len(t)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    tau = t[1] - t[0]
    h = x[1] - x[0]
    print('tau', tau)
    print('h', h)
    print(check(h, tau))
    beta = alpha * tau / (2 * C * h ** 2)
    mu = 1 + 2 * beta + D * tau / (2 * C)
    p = []
    args.x -= 2
    for i in range(0, args.x):
        line = []
        if (i == 0):
            line.append(mu)
            line.append(-beta)
            for j in range(2, args.x):
                line.append(0)
        elif (i == args.x):
            for j in range(0, args.x - 2):
                line.append(0)
            line.append(-beta)
            line.append(mu)
        else:
            for j in range(0,args.x):
                if (j == i - 1):
                    line.append(-beta)
                elif (j == i):
                    line.append(mu)
                elif (j == i + 1):
                    line.append(-beta)
                else:
                    line.append(0)

        p.append(line)
    p = np.array(p)

    q = np.zeros(args.x)
    uprev = np.zeros(args.x)
    #print(q)
    err = [0]*n
    for i in range(n-1):
        q[0] = beta * (phi(t[i+1]) + phi(t[i]))
        q[-1] = beta * phi(t[i+1] + phi(t[i]))
        rightPart = np.zeros(args.x)
        rightPart[0] = beta * uprev[1] + uprev[0] * (1 - 2 * beta - D * tau / (2 * C)) 
        rightPart[-1] = beta * uprev[-2] + uprev[-1] * (1 - 2 * beta - D * tau / (2 * C)) 

        for j in range(1, args.x-1):
            rightPart[j] = beta * uprev[j - 1] + beta * uprev[j + 1] + (1 - 2 * beta - D * tau / (2 * C)) * uprev[j]

        rightPart += q
        u_num = np.linalg.solve(p, rightPart)

        uprev = u_num
        # eps = []
        # for j in range(len(p)):
        #     eps.append(np.dot(p[j], u_num)- rightPart[j])
        # print(max(eps))
        data = u(x, t[i+1], num_steps)
        # logging.info(f'At plotting step {i + 1} out of {n}, error: {np.linalg.norm(u_num - data)}')
        forplot = [phi(t[i+1])] + list(u_num) + [phi(t[i+1])]
        forplot = np.array(list(map(float, forplot)))
        data = np.array(list(map(float, data)))
        localerr = np.abs(forplot - data)
        err[i] = np.max(np.abs(forplot - data))
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.xlim([-1, l + 1])
        plt.ylim([-.05, .25])
        ax.plot(x, data, label=f'u(x) at time {round(t[i], 2)}')
        ax.plot(x, forplot, label=f'u_num(x) at time {round(t[i], 2)}')
        ax.plot([-1, l + 1], [0, 0])
        ax.plot([-1, l + 1], [phi(t[i+1]), phi(t[i+1])], label='phi(t)')
        ax.legend()
        fig.savefig('plots/plot{:03}.png'.format(i))
        plt.close(fig)
    # max_ind = err.index(max(err))
    # err[max_ind] = 0
    max_ind = err.index(max(err))
    max_err = err[max_ind]
    print(max_err)
