import equations.numerical_solution as num
import equations.analytic_solution as an
import matplotlib.pyplot as plt


if __name__ == '__main__':
    time = 4
    # an.static_2d(time)
    # num.static_2d(time)
    # an.animated_2d(time)
    num.animated_2d(time)
    plt.show()