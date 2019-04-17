import numpy as np
import timeit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

# Constants
LX = 4
LY = 1
A = 1

# Number of grid nodes per time and space variables
K_big = 100
I_big = 100

ANIMATION_TIME_STEPS = 100


def __plot_2d(x_sp, y_sp, t, figname, vline=0, y=0, savefig=False):
    fig = plt.figure(figname)
    ax = plt.subplot(111)

    plt.rc('lines', linewidth=1)

    graph, = ax.plot(x_sp, y_sp, color='orange', marker='o',
                     linestyle='-', linewidth=2, markersize=0.1)

    plt.xlabel('x')
    plt.ylabel('z')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    if vline != 0:
        line = plt.axvline(x=vline, color='r')
        ax.legend([line, graph], ['x={0}'.format(vline), 'u(x,y,t) at t={0}'.format(t)],
                  loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)
    else:
        ax.legend([graph], ['numeric solution at t={0}'.format(t)],
                  loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    if savefig:
        name = '{0}_{1}_{2}'.format(vline, y, t).replace('.', '')
        plt.savefig(name)
    # plt.show()


def __anim_plot_2d(x_vals, y_per_time, h_t):
    fig = plt.figure("Numerical solution animated")
    ax = plt.axes(xlim=(0, LX), ylim=(- 1.5, 1.5))
    line, = ax.plot([], [], lw=2, color="orange")
    time_text = ax.text(.2, 1.5, '', fontsize=15)

    plt.xlabel('x')
    plt.ylabel('y')

    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        line.set_data([], [])
        return line, time_text

    # animation function.  This is called sequentially
    def animate(i):
        index = i % len(y_per_time)
        x = np.linspace(0, LX, I_big)
        y = y_per_time[index]
        line.set_data(x, y)
        time_text.set_text('T={0}'.format(round(index * h_t, 2)))
        return line, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=200, interval=30, blit=False)
    plt.show()


def get_value_at(x, time):
    h_x = LX / I_big
    h_t = time / K_big
    res = differential_scheme(h_x, h_t)

    for i in range(len(res)-1):
        if i * h_x == x:
            return res[i]

        if i*h_x < x < (i+1)*h_x:
            left_x = h_x * i
            t = (x-left_x) / h_x
            value = left_x + t * h_x
            return value


def static_2d(time, figname="Numerical solution"):
    start = timeit.default_timer()
    print("Starting calculation of numerical solution...")

    h_x = LX / I_big
    h_t = time / K_big

    print("h_x = ", h_x, "; h_t = ", h_t)
    res = differential_scheme(h_x, h_t)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __plot_2d(np.linspace(0, LX, I_big), res, time, figname)


def animated_2d(time):
    t_vals = np.linspace(0, time, ANIMATION_TIME_STEPS)

    x_vals = np.linspace(0, I_big, I_big)

    values_per_time = []

    start = timeit.default_timer()
    print("Starting calculation for animated 2d...")
    time_step = time / ANIMATION_TIME_STEPS

    h_x = LX / I_big
    for t in t_vals:
        h_t = t / K_big
        res = differential_scheme(h_x, h_t)
        values_per_time.append(res)
    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __anim_plot_2d(x_vals, values_per_time, time_step)


# Initial shape
def psi(x):
    return -(x ** 2) / LX + x


# "gamma" factor equals (a*h_t/h_x)^2
def gamma_func(h_t, h_x):
    return (A * h_t / h_x) ** 2


# Solution of a differential scheme (v_i_k+1) for given indices
def solve_k_plus1(gamma, h_t: float, i: int, v_k: list, v_k_minus1: list):
    return gamma * v_k[i - 1] + (-2 * gamma + 2 - (A * np.pi * h_t / LY) ** 2) * v_k[i] + gamma \
           * v_k[i + 1] - v_k_minus1[i]


# Full solution of a differential scheme
def differential_scheme(h_x, h_t):
    # Grid of a differentiol scheme
    grid = []
    for k in range(K_big):
        grid.append([0] * I_big)

    # _____________________________________________________
    # Setting the initial shape (v(k=0))
    x = np.linspace(0, LX, I_big)
    for i in range(0, I_big):
        grid[0][i] = psi(x[i])

    # Values at v(k=1) are equal to v(k=0)
    # v_i_k = v_i_k_minus1
    # _____________________________________________________

    # return grid[0]
    gamma = gamma_func(h_t, h_x)
    grid[1] = grid[0]
    # Computing full solution with given amount of time steps
    for k in range(1, K_big - 1):
        grid[k + 1][0] = 0
        # print(grid[k][int(I_big / 2)])
        for i in range(1, I_big - 1):
            grid[k + 1][i] = solve_k_plus1(gamma, h_t, i, grid[k], grid[k - 1])
        grid[k + 1][-1] = 0

    return grid[-1]


if __name__ == '__main__':
    pass
    # static_2d(2)
    # animated_2d(2)
