from functools import lru_cache

import fire
import numpy
from matplotlib import animation
from matplotlib import pyplot as plt
from mpmath import quad, legendre
from sympy import Symbol, legendre_poly


@lru_cache(maxsize=123)
def legendre_p(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x


def lam_n(n, k, a, l):
    return k * (n ** 2 + n + (a / (k * l)))


def draw_graphics(variant, const_k, const_c, const_a, const_l, frames, fps):
    def animate(val):
        time = val / 10

        def super_exp(n):
            return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

        x_s = []
        y_s = []
        for j in numpy.arange(-1, 1, 0.01):
            if variant == 4:
                summary = (1 / 5) * super_exp(0) + (2 / 7) * (3 * (j ** 2) - 1) * super_exp(1) + (1 / 35) * (
                        35 * (j ** 4) - 30 * (j ** 2) + 3) * super_exp(4)
            else:
                summary = (27 / 63) * j * super_exp(1) + (14 / 63) * (5 * (j ** 3) - 3 * j) * super_exp(3) + (
                        1 / 63) * (63 * (j ** 5) - 70 * (j ** 3) + 15 * j) * super_exp(5)
            x_s.append(j)
            y_s.append(summary)
        ax1.clear()
        ax1.set_xlabel("Z")
        ax1.set_ylabel("$\omega(z, {})$".format(time))
        ax1.plot(x_s, y_s)

    plt.rc('font', **{'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    anim = animation.FuncAnimation(fig, animate, interval=1, frames=frames)

    anim.save('variant-time-{}.mp4'.format(variant), fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()


def draw_graphics2(variant, const_k, const_c, const_a, const_l, frames, fps, u_c):
    def animate(val):
        time = val / 10

        def super_exp(n):
            return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

        x_s = []
        y_s = []
        for j in numpy.arange(0, numpy.pi, 0.01):
            if variant == 4:
                summary = (1 / 5) * super_exp(0) + (2 / 7) * (3 * (numpy.cos(j) ** 2) - 1) * super_exp(1) + (1 / 35) * (
                        35 * (numpy.cos(j) ** 4) - 30 * (numpy.cos(j) ** 2) + 3) * super_exp(4)
            else:
                summary = (27 / 63) * numpy.cos(j) * super_exp(1) + (14 / 63) * (
                        5 * (numpy.cos(j) ** 3) - 3 * numpy.cos(j)) * super_exp(3) + (
                                  1 / 63) * (63 * (numpy.cos(j) ** 5) - 70 * (numpy.cos(j) ** 3) + 15 * numpy.cos(
                    j)) * super_exp(5)
            x_s.append(j)
            y_s.append(summary + u_c)
        ax1.clear()
        plt.grid()
        ax1.set_xlabel("$\\theta$")
        ax1.set_ylabel("$\omega(\\theta, {})$".format(time))
        ax1.plot(x_s, y_s)

    plt.rc('font', **{'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    anim = animation.FuncAnimation(fig, animate, interval=1, frames=frames)

    anim.save("variant-time-{}.mp4".format(variant), fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()


def draw_graphics_by_z(variant, const_k, const_c, const_a, const_l, frames, fps):
    def animate(val):
        z = val % 100
        if z < 50:
            z = 50 - z
            z *= -1
        elif z == 50:
            z = 0
        else:
            z -= 50
        z /= 50

        def super_exp(n, time):
            return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

        x_s = []
        y_s = []
        for j in numpy.arange(0, 20, 0.01):
            if variant == 4:
                summary = (1 / 5) * super_exp(0, j) + (2 / 7) * (3 * (z ** 2) - 1) * super_exp(1, j) + (1 / 35) * (
                        35 * (z ** 4) - 30 * (z ** 2) + 3) * super_exp(4, j)
            else:
                summary = (27 / 63) * z * super_exp(1, j) + (14 / 63) * (5 * (z ** 3) - 3 * z) * super_exp(3, j) + (
                        1 / 63) * (63 * (z ** 5) - 70 * (z ** 3) + 15 * z) * super_exp(5, j)
            x_s.append(j)
            y_s.append(summary)
        ax1.clear()
        ax1.set_xlabel("t")
        ax1.set_ylabel("$\omega({}, t)$".format(z))
        ax1.plot(x_s, y_s)

    plt.rc('font', **{'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    anim = animation.FuncAnimation(fig, animate, interval=1, frames=frames)

    anim.save('variant-z-{}.mp4'.format(variant), fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()


def draw_graphics_by_z2(variant, const_k, const_c, const_a, const_l, frames, fps, u_c):
    def animate(val):
        theta = (numpy.pi * val) / frames

        def super_exp(n, time):
            return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

        x_s = []
        y_s = []
        for j in numpy.arange(0, 20, 0.01):
            if variant == 4:
                summary = (1 / 5) * super_exp(0, j) + (2 / 7) * (3 * (numpy.cos(theta) ** 2) - 1) * super_exp(1, j) + (
                        1 / 35) * (
                                  35 * (numpy.cos(theta) ** 4) - 30 * (numpy.cos(theta) ** 2) + 3) * super_exp(4, j)
            else:
                summary = (27 / 63) * numpy.cos(theta) * super_exp(1, j) + (14 / 63) * (
                        5 * (numpy.cos(theta) ** 3) - 3 * numpy.cos(theta)) * super_exp(3, j) + (
                                  1 / 63) * (
                                  63 * (numpy.cos(theta) ** 5) - 70 * (numpy.cos(theta) ** 3) + 15 * numpy.cos(
                              theta)) * super_exp(5, j)
            x_s.append(j)
            y_s.append(summary + u_c)
        ax1.clear()
        ax1.set_xlabel("t")
        ax1.set_ylabel("$\omega({}, t)$".format(theta))
        ax1.plot(x_s, y_s)

    plt.rc('font', **{'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    anim = animation.FuncAnimation(fig, animate, interval=1, frames=frames)

    anim.save('variant-z-{}.mp4'.format(variant), fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.show()


def show_image(const_k, const_c, const_a, const_l, x_s, y_s, label_x, label_y):
    plt.figure()
    plt.grid()
    plt.gca().set_position((.1, .3, .8, .6))
    plt.rc('font', **{'serif': ['Computer Modern']})
    plt.rc('text', usetex=False)
    plt.plot(y_s, x_s)
    plt.xlabel(label_y)
    plt.ylabel(label_x)
    plt.figtext(
        .0, .0,
        "  Chosen Parameters:\n  $alpha$ = {0}\n  c = {1}\n  k = {2}\n  l = {3}\n".format(const_a, const_c, const_k, const_l)
    )
    plt.show()


def show_by_time(variant, const_k, const_c, const_a, const_l, time):
    def super_exp(n):
        return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

    x_s = []
    y_s = []
    for j in numpy.arange(-1, 1, 0.01):
        if variant == 4:
            summary = (1 / 5) * super_exp(0) + (2 / 7) * (3 * (j ** 2) - 1) * super_exp(1) + (1 / 35) * (
                    35 * (j ** 4) - 30 * (j ** 2) + 3) * super_exp(4)
        else:
            summary = (27 / 63) * j * super_exp(1) + (14 / 63) * (5 * (j ** 3) - 3 * j) * super_exp(3) + (
                    1 / 63) * (63 * (j ** 5) - 70 * (j ** 3) + 15 * j) * super_exp(5)
        x_s.append(j)
        y_s.append(summary)
    show_image(const_k, const_c, const_a, const_l, x_s, y_s, "z", "$\omega(z, t)$")


def show_by_time2(variant, const_k, const_c, const_a, const_l, time, u_c):
    def super_exp(n):
        return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

    x_s = []
    y_s = []
    for j in numpy.arange(0, numpy.pi, 0.01):
        if variant == 4:
            summary = (1 / 5) * super_exp(0) + (2 / 7) * (3 * (numpy.cos(j) ** 2) - 1) * super_exp(1) + (1 / 35) * (
                    35 * (numpy.cos(j) ** 4) - 30 * (numpy.cos(j) ** 2) + 3) * super_exp(4)
        else:
            summary = (27 / 63) * numpy.cos(j) * super_exp(1) + (14 / 63) * (
                    5 * (numpy.cos(j) ** 3) - 3 * numpy.cos(j)) * super_exp(3) + (
                              1 / 63) * (
                              63 * (numpy.cos(j) ** 5) - 70 * (numpy.cos(j) ** 3) + 15 * numpy.cos(j)
                      ) * super_exp(5)
        x_s.append(summary + u_c)
        y_s.append(j)
    show_image(const_k, const_c, const_a, const_l, x_s, y_s, "$\omega(\\theta, {})$".format(time), "$\\theta$")


def show_by_z(variant, const_k, const_c, const_a, const_l, z):
    def super_exp(n, time):
        return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

    x_s = []
    y_s = []
    for j in numpy.arange(0, 20, 0.1):
        if variant == 4:
            summary = (1 / 5) * super_exp(0, j) + (2 / 7) * (3 * (z ** 2) - 1) * super_exp(1, j) + (1 / 35) * (
                    35 * (z ** 4) - 30 * (z ** 2) + 3) * super_exp(4, j)
        else:
            summary = (27 / 63) * z * super_exp(1, j) + (14 / 63) * (5 * (z ** 3) - 3 * z) * super_exp(3, j) + (
                    1 / 63) * (63 * (z ** 5) - 70 * (z ** 3) + 15 * z) * super_exp(5, j)
        x_s.append(summary)
        y_s.append(j)
    show_image(const_k, const_c, const_a, const_l, x_s, y_s, "$\omega({}, t)$".format(z), "t")


def show_by_z2(variant, const_k, const_c, const_a, const_l, theta, u_c):
    def super_exp(n, time):
        return numpy.exp(-lam_n(n, const_k, const_a, const_l) * time / const_c)

    x_s = []
    y_s = []
    for j in numpy.arange(0, 20, 0.1):
        if variant == 4:
            summary = (1 / 5) * super_exp(0, j) + (2 / 7) * (3 * numpy.cos(theta) ** 2 - 1) * super_exp(1, j) + (
                    1 / 35) * (
                              35 * (numpy.cos(theta) ** 4) - 30 * (numpy.cos(theta) ** 2) + 3) * super_exp(4, j)
        else:
            summary = (27 / 63) * numpy.cos(theta) * super_exp(1, j) + (14 / 63) * (
                    5 * (numpy.cos(theta) ** 3) - 3 * numpy.cos(theta)) * super_exp(3, j) + (
                              1 / 63) * (
                              63 * (numpy.cos(theta) ** 5) - 70 * (numpy.cos(theta) ** 3) + 15 * numpy.cos(
                          theta)) * super_exp(5, j)
        x_s.append(summary + u_c)
        y_s.append(j)
    show_image(const_k, const_c, const_a, const_l, x_s, y_s, "$\omega({}, t)$".format(theta), "t")


def get_A_n(variant, n, accuracy):
    return (quad(lambda x: legendre(n, x) * x ** variant, [-1, 1], method='gauss-legendre', maxdegree=accuracy) * (
            (2 * n) + 1)) / 2


def find_const(variant, count, accuracy):
    A_ns = []
    for i in range(count):
        A_n = get_A_n(variant, i, accuracy)
        print("A_{0} = {1}".format(i, A_n))
        A_ns.append(A_n)
    return A_ns


def calculate_series(z, t, variant, const_k, const_c, const_a, const_t, const_l, count):
    def super_exp(n):
        return numpy.exp(-lam_n(n, const_k, const_a, const_l) * const_t / const_c)

    z = Symbol('z')
    for i in range(count):
        poly = legendre_poly(i, z)
        sum += (2 * i) * super_exp(i) * poly * get_A_n(variant, i)


class CLI(object):
    """
    To show graphics call "draw" as argument of program.
    To calculate $A_n$ call "find-a" as argument of program.
    To calculate $\omega(z,t)$ series, call "series" as argument of program.
    """

    def draw(self, power=4, k=0.59, c=1.65, a=2e-3, l=0.5, u_c=18, frames=200, fps=60, gtype='time'):
        """
        Function to visualize heat process.
        :param power: Power of given function. $\psi$. $\psi(z) = z^{\text{power}}$
        :param k: Thermal conductivity coefficient.
        :param c: bulk heat capacity.
        :param a: heat transfer coefficient.
        :param l: shell thickness.
        :param frames: The number of frames of animation.
        :param fps: Frames per second.
        """
        print("-" * 20)
        print("Chosen params:")
        print('k = {}'.format(k))
        print('c = {}'.format(c))
        print('alpha = {}'.format(a))
        print('l = {}'.format(l))
        print('frames will be shown: {}'.format(frames))
        print('Frames per second: {}'.format(fps))
        if gtype == 'time':
            draw_graphics2(int(power), float(k), float(c), float(a), float(l), int(frames), int(fps), float(u_c))
        else:
            draw_graphics_by_z2(int(power), float(k), float(c), float(a), float(l), int(frames), int(fps), float(u_c))

    def find_a(self, power=4, count=5, accuracy=2):
        """
        Function will calculate $A_n$ for chosen power.
        :param power: Power of given function. $\psi$. $\psi(z) = z^{\text{power}}$
        :param count: Quantity of $A_n$. $A_0$ ... $A_{\text{count}}$ will be calculated.
        :param accuracy: Calculus accuracy.
        """
        find_const(int(power), count, accuracy)

    def pic(self, power=4, k=0.59, c=1.65, a=2e-3, l=0.5, time=20, z=0, gtype='time', u_c=18):
        """
        Function to show picture of $\omega(z,t)$ with $t$=time
        :param gtype: type of graphic. Possible values: 'time' and 'z'.
        :param z: z value
        :param power: Power of given function. $\psi$. $\psi(z) = z^{\text{power}}$
        :param k: Thermal conductivity coefficient.
        :param c: bulk heat capacity.
        :param a: heat transfer coefficient.
        :param l: shell thickness.
        :param time: time parameter.
        :param u_c: Environment temperature.
        """
        print("-" * 20)
        print("Chosen params:")
        print('k = {}'.format(k))
        print('c = {}'.format(c))
        print('alpha = {}'.format(a))
        print('l = {}'.format(l))
        if gtype == 'time':
            show_by_time2(int(power), float(k), float(c), float(a), float(l), float(time), float(u_c))
        else:
            show_by_z2(int(power), float(k), float(c), float(a), float(l), float(z), float(u_c))

    def series(self, power=4, k=0.59, c=1.65, a=2e-3, l=0.5, t=20, count=5):
        """
        Function to calculate series $\omega(z,t)|_{t=0} = \sum_{n=0}^{\text{count}}{A_n P_n(z)}$
        :param power: Power of given function. $\psi$. $\psi(z) = z^{\text{power}}$
        :param count: Series elements quantity.
        :param k: Thermal conductivity coefficient.
        :param c: bulk heat capacity.
        :param a: heat transfer coefficient.
        :param l: shell thickness.
        """
        calculate_series(int(power), float(k), float(c), float(a), float(t), float(l), float(count))


if __name__ == "__main__":
    fire.Fire(CLI)
