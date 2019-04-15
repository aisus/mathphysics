import numpy as np
import timeit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

T_MIN = 0
N_MAX = 20
GRID_STEP = 0.04
TIME_STEP = 0.1
LX = 4
LY = 1


def __plot_2d(x_sp, y_sp, t, vline=0, y=0, savefig=False):
    fig = plt.figure()
    ax = plt.subplot(111)

    plt.rc('lines', linewidth=1)

    graph, = ax.plot(x_sp, y_sp, color='b', marker='o',
                     linestyle='-', linewidth=2, markersize=0.1)

    plt.xlabel('x')
    plt.ylabel('y')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # if vline != 0:
    #     line = plt.axvline(x=vline, color='r')
    #     ax.legend([line, graph], ['x={0}'.format(vline), 'u(x,y,t) at t={0}'.format(t)],
    #               loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)
    # else:
    ax.legend([graph], ['u(x,y,t) at t={0}'.format(t)],
                loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    if savefig:
        name = '{0}_{1}_{2}'.format(vline, y, t).replace('.', '')
        plt.savefig(name)
    plt.show()


def __anim_plot_2d(x_vals, y_per_time):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, LX), ylim=(-LY * 2, LY * 2))
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
        time_text.set_text('T={0}'.format(round(index * TIME_STEP, 3)))
        return line, time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=200, interval=60, blit=False)
    plt.show()


def __plot_3d(t, x, y, z):
    fig = plt.figure()
    axes = Axes3D(fig)

    axes.plot_surface(x, y, z, cmap='inferno')

    plt.show()
    name = '{0}'.format(t).replace('.', '')
    plt.savefig('{0}_3d'.format(name))

def __anim_plot_3d(x_vals, y_vals, z_per_time):
    fig = plt.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x_vals, y_vals, z_per_time[0])

    ##time_text = axes.text(.5, .5, '', fontsize=15, )

    # animation function.  This is called sequentially
    def animate(i):
        axes.clear()

        axes.autoscale(False, axis='z', tight=None)
        axes.set_zlim(-1, 1)

        index = i % len(z_per_time)
        z = z_per_time[index]

        # time_text.set_text('T={0}'.format(round(index*TIME_STEP,3)))
        return axes.plot_surface(x_vals, y_vals, z, cmap='inferno'),  # time_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=200, interval=60, blit=False)

    plt.show()


def __c_n(n):
    return 8 * (np.pi * n * np.sin(np.pi * n) + 2 * np.cos(np.pi * n) - 2) / (np.pi ** 3 * n ** 3)


def __series_element(n, x, t):
    return np.sin(n * np.pi * x / 4) * __c_n(n) * np.cos(np.sqrt((n * np.pi / 4) ** 2 + np.pi ** 2) * t)


def __series_sum_2d(N, x, t):
    res = 0
    for n in range(1, N):
        res += -__series_element(n, x, t)
    return res


def __series_sum_3d(N, x, y, t):
    res = 0
    for n in range(1, N):
        res += -__series_element(n, x, t)
    return res * np.sin(np.pi * y / LY)


def static_2d(time):
    x = np.linspace(0, LX, int(LX / GRID_STEP))

    start = timeit.default_timer()
    print("Starting calculation for 2d...")

    res = __series_sum_2d(N_MAX, x, time)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __plot_2d(x, res, time)


def animated_2d(time, n=N_MAX):
    time_slices_count = int((time - T_MIN) / TIME_STEP)
    T = np.linspace(T_MIN, time, time_slices_count)

    grid_points_count = int(LX / GRID_STEP)
    x_vals = np.linspace(0, LX, grid_points_count)

    values_per_time = []

    start = timeit.default_timer()
    print("Starting calculation for animated 2d...")
    for t in T:
        res = []
        for x in x_vals:
            subres = __series_sum_2d(n, x, t)
            res.append(subres)
        values_per_time.append(res)
    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __anim_plot_2d(x_vals, values_per_time)


def animated_3d(time, n=N_MAX):
    time_slices_count = int((time - T_MIN) / TIME_STEP)
    T = np.linspace(T_MIN, time, time_slices_count)

    x = np.linspace(0, LX, int(LX / GRID_STEP))
    y = np.linspace(0, LY, int(LY / GRID_STEP))

    start = timeit.default_timer()
    print("Starting calculation for animated 3d...")

    xv, yv = np.meshgrid(x, y)
    values_per_time = []

    for t in T:
        res = __series_sum_3d(n, xv, yv, t)
        values_per_time.append(res)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __anim_plot_3d(xv, yv, values_per_time)


def static_3d(time, n=N_MAX):
    x = np.linspace(0, LX, int(LX / GRID_STEP))
    y = np.linspace(0, LY, int(LY / GRID_STEP))

    start = timeit.default_timer()
    print("Starting calculation for 3d...")

    xv, yv = np.meshgrid(x, y)

    res = __series_sum_3d(n, xv, yv, time)

    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(end - start))

    __plot_3d(time, xv, yv, res)


def calculate_with_precision(x, y, t, eps):
    x_sp = np.linspace(0, LX, int(LX / GRID_STEP))
    n = int(find_n(eps))
    n = 500

    start = timeit.default_timer()
    print("Starting calculation with given precision {0}, N is {1}".format(eps, n))
    res = __series_sum_3d(n, x, y, t)
    end = timeit.default_timer()
    print("Finished calculation in {0}s".format(round(end - start, 5)))
    print("Result: {0}".format(res))

    y_sp = __series_sum_3d(n, x_sp, y, t)

    __plot_2d(x_sp, y_sp, t, x, y, True)


def find_n(eps):
    ##return (4 * np.sqrt(np.pi*eps+1)) / (np.pi**2 * eps)
    return (4 * np.sqrt(np.pi * eps + 1) + 1) / ((np.pi ** 2) * eps)


def test_precision(t, eps, n_step):
    x = 2
    y = 0.5
    x_sp = np.linspace(0, LX, int(LX / GRID_STEP))
    n = int(find_n(eps))

    print("Starting test for ", eps)
    print("Theor.n is ", n)
    print("max number of calcs is", n / n_step)
    n = 600
    prev_res = __series_sum_3d(n, x, y, t)
    while True:
        res = __series_sum_3d(n, x, y, t)
        if abs(res - prev_res) < eps and n >= 0:
            prev_res = res
            n -= n_step
            print('n: ', n, ' d: ', abs(res - prev_res))
            continue
        else:
            print("found n as ", n, " for eps=", eps)
            break


if __name__ == '__main__':
    test_precision(10000, 0.1, 1)
