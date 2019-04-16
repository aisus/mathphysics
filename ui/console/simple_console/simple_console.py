import equations.analytic_solution as analytic
import equations.numerical_solution as numeric
import matplotlib.pyplot as plt


def display_man():
    print('=========== Manual ================')
    print('Type \'-man\' to display this manual')
    print('To compute equation, type:')
    print('\'-calc x y t eps\'')
    print('where x, y - coordinates of point,')
    print('t - time in seconds, eps - precision')
    print('To show graph at a given moment, type:')
    print('\'-g2 t\' for 2d')
    print('\'-g3 t\' for 3d')
    print('To show animated graph form 0 to a given moment, type:')
    print('\'-a2 t\' for 2d')
    print('\'-a3 t\' for 3d')
    print('===================================')


def input_loop():
    while True:
        user_input = input().split()

        if user_input[0] == '-man':
            display_man()
            continue
        if user_input[0] == '-a2':
            analytic.animated_2d(float(user_input[1]))
        elif user_input[0] == '-g2':
            analytic.static_2d(float(user_input[1]))
        elif user_input[0] == '-g3':
            analytic.static_3d(float(user_input[1]))
        elif user_input[0] == '-a3':
            analytic.animated_3d(float(user_input[1]))
        elif user_input[0] == '-calc':
            analytic.calculate_with_precision(float(user_input[1]), float(user_input[2]), float(user_input[3]),
                                     float(user_input[4]))
        elif user_input[0] == '-both':
            time = float(user_input[1])
            analytic.static_2d(time, "Solution")
            numeric.static_2d(time, "Solution")
            plt.show()
        else:
            print("invalid command")


if __name__ == '__main__':
    print('\n********************************')
    print('* MEMBRANE EQUATION CALCULATOR *')
    print('********************************')
    display_man()
    input_loop()
