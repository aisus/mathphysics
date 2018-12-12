from equations import membrane


def display_man():
    print('=========== Manual ===========')


def input_loop():
    while True:
        user_input = input().split()
        if user_input[0] == '2d':
            membrane.animated_2d(float(user_input[1]))
        elif user_input[0] == '3d':
            membrane.static_3d(float(user_input[1]))
        elif user_input[0] == 'a3d':
            membrane.animated_3d(float(user_input[1]))


if __name__ == '__main__':
    print('\n********************************')
    print('* MEMBRANE EQUATION CALCULATOR *')
    print('********************************')
    input_loop()
