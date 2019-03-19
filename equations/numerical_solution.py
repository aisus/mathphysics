import numpy as np
import timeit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

# Constants
T = 10
LX = 4
LY = 1
A = 1

# Number of grid nodes per time and space variables
K_big = 50
I_big = 50

ANIMATION_TIME_STEPS = 100


def __plot_2d(x_sp, y_sp, t, vline=0, y=0, savefig=False):
    fig = plt.figure()
    ax = plt.subplot(111)

    plt.rc('lines', linewidth=1)

    graph, = ax.plot(x_sp, y_sp, color='b', marker='o',
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
        ax.legend([graph], ['u(x,y,t) at t={0}'.format(t)],
                  loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    if savefig:
        name = '{0}_{1}_{2}'.format(vline, y, t).replace('.', '')
        plt.savefig(name)
    plt.show()


def __anim_plot_2d(x_vals, y_per_time):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, I_big), ylim=(- 4, 4))
    line, = ax.plot([], [], lw=2)
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
        x = x_vals
        y = y_per_time[index]
        line.set_data(x, y)
        time_text.set_text('k={0}'.format(round(index, 3)))
        return line, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=200, interval=60, blit=False)
    plt.show()


def static_2d(time):
    x = np.linspace(0, I_big + 1, I_big + 1)

    start = timeit.default_timer()
    print("Starting calculation of numerical solution...")

    res = differential_scheme_full(time)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __plot_2d(x, res, time)


def animated_2d(time):
    t_vals = np.linspace(0, time, ANIMATION_TIME_STEPS)

    x_vals = np.linspace(0, I_big + 1, I_big + 1)

    values_per_time = []

    start = timeit.default_timer()
    print("Starting calculation for animated 2d...")
    for t in t_vals:
        res = differential_scheme_full(t)
        values_per_time.append(res)
    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __anim_plot_2d(x_vals, values_per_time)


# Solution of a differential scheme
def differential_scheme_full(time):
    # Grid step per space variable
    h_x = LX / I_big
    # Grid step per time variable
    h_t = time / K_big

    v_i_k_minus1 = np.zeros(I_big + 1)
    v_i_k = np.zeros(I_big + 1)
    v_i_k_plus1 = np.zeros(I_big + 1)

    # _____________________________________________________
    # Setting the initial shape
    gamma = (A * h_t / h_x) ** 2

    for i in range(0, I_big):
        v_i_k_minus1[i] = (- ((i * h_x) ** 2) / 4 + i * h_x)

    # return v_i_k_minus1

    # Setting the first time step
    v_i_k[0] = 0
    for i in range(1, I_big - 1):
        v_i_k[i] = gamma * v_i_k_minus1[i - 1] + gamma * v_i_k_minus1[i + 1] + (
                2 - 2 * gamma - (A * h_t * np.pi / LY) ** 2) * v_i_k_minus1[i]
    v_i_k[I_big] = 0

    # _____________________________________________________

    for k in range(1, K_big-1):
        v_i_k_plus1[0] = 0
        for i in range(1, I_big - 1):
            v_i_k_plus1[i] = gamma * v_i_k[i - 1] + gamma * v_i_k[i + 1] + (
                    2 - 2 * gamma - (A * h_t * np.pi / LY) ** 2) * v_i_k[i] - v_i_k_minus1[i]
        v_i_k_plus1[I_big] = 0
        v_i_k_minus1 = v_i_k
        v_i_k = v_i_k_plus1

    return v_i_k_plus1


if __name__ == '__main__':
    animated_2d(4)
