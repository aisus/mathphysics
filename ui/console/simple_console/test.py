import equations.numerical_solution as num
import equations.analytic_solution as an
import matplotlib.pyplot as plt

if __name__ == '__main__':
    time = 1.5
    x = 1
    num.I_big = 500
    num.K_big = 500
    # an.static_2d(time)
    num.static_2d(time)
    # an.animated_2d(time)
    # num.animated_2d(time)

    # an.static_2d(time, 'solution')
    # num.static_2d(time, 'solution')

    print(num.get_value_at(x, time))
    # print(an.get_value_at(x, time))

    plt.show()
