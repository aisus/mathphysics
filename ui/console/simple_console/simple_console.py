from equations.membrane import *


def display_man():
    print('=========== Manual ===========')


def input_loop():
    while True:
        user_input = input().split()
        if user_input[0] == 'a2d':
            animated_2d(float(user_input[1]))
        elif user_input[0] == '2d':
            static_2d(float(user_input[1]))
        elif user_input[0] == '3d':
            static_3d(float(user_input[1]))
        elif user_input[0] == 'a3d':
            animated_3d(float(user_input[1]))


if __name__ == '__main__':
    print('\n********************************')
    print('* MEMBRANE EQUATION CALCULATOR *')
    print('********************************')
    input_loop()
