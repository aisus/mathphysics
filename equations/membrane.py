import numpy as np
import pylab
import timeit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

T_MIN = 0
T_MAX = 10
N_MAX = 20
GRID_STEP = 0.04
TIME_STEP = 0.1
LX = 4
LY = 1


##____________________GRAPHS_________________________
def graph_2d(x, y, name, color):
    fig = plt.figure()
    ax = plt.subplot(111)

    plt.rc('lines', linewidth=1)

    graph, = ax.plot(x, y, color=color, marker='o',
                     linestyle='-', linewidth=2, markersize=0.1)

    plt.xlabel('x')
    plt.ylabel('y')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend([graph], [name],
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    # plt.savefig(name + '.png')
    plt.show()


def anim_graph_2d(x_vals, y_per_time):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, LX), ylim=(-LY * 2, LY * 2))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        index = i % len(y_per_time)
        x = x_vals
        y = y_per_time[index]
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=60, blit=True)

    plt.show()

def graph_3d(x, y, z):
    fig = pylab.figure()
    axes = Axes3D(fig)

    axes.plot_surface(x, y, z)

    pylab.show()


def anim_graph_3d():
    pass


##___________________________________________________

##___________________SERIES_SUM______________________
def c_n(n):
    return 8 * (np.pi * n * np.sin(np.pi * n) + 2 * np.cos(np.pi * n) - 2) / (np.pi ** 3 * n ** 3)


def series_subsum(n, x, t):
    return np.sin(n * np.pi * x / 4) * c_n(n) * np.cos(np.sqrt((n * np.pi / 4) ** 2 + (np.pi) ** 2) * t)


def series_sum(N, x, t):
    res = 0
    for n in range(1, N):
        res += -series_subsum(n, x, t)
    return res


##___________________________________________________

def main_2d():
    time_slices_count = int((T_MAX - T_MIN) / TIME_STEP)
    T = np.linspace(T_MIN, T_MAX, time_slices_count)

    grid_points_count = int(LX / GRID_STEP)
    x_vals = np.linspace(0, LX, grid_points_count)

    values_per_time = []

    start = timeit.default_timer()
    print("Starting calculation for 2d...")
    for t in T:
        res = []
        for x in x_vals:
            subres = series_sum(N_MAX, x, t)
            res.append(subres)
        values_per_time.append(res)
    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    ## anim_graph_2d()
    return x_vals, values_per_time


def main_3d():
    x_vals = np.linspace(0, LX, int(LX / GRID_STEP))
    y_vals = np.linspace(0, LY, int(LY / GRID_STEP))

    start = timeit.default_timer()
    print("Starting calculation for 2d...")

    t = 4
    res = []
    for x in x_vals:
        subres = series_sum(N_MAX, x, t)
        res.append(subres)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))


if __name__ == '__main__':
    main_2d()
