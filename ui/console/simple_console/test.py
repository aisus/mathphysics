import equations.numerical_solution as num
import equations.analytic_solution as an
import matplotlib.pyplot as plt


def to_file(filename, i_s: list, k_s: list, ht_s: list, hx_s: list, eps: list, deltas: list):
    f = open(filename, 'w+')
    f.write("I    K    ht    hx    eps    d\n")
    for i in range(len(i_s)):
        f.write(f"{i_s[i]}    {k_s[i]}    {ht_s[i]}    {hx_s[i]}    {eps[i]}    {deltas[i]}\n")


if __name__ == '__main__':
    time = 2
    x = 1
    num.I_big = 500
    num.K_big = 500
    # an.static_2d(time)
    # num.static_2d(time)
    # an.animated_2d(time)
    # num.animated_2d(time)

    # an.static_2d(time, 'solution')
    # num.static_2d(time, 'solution')

    #print(num.get_value_at(x, time))
    # print(an.get_value_at(x, time))
    #plt.show()

    i_big = 4
    k_big = 4
    eps = []
    i_s = []
    k_s = []
    ht_s = []
    hx_s = []
    deltas = []
    for i in range(6):
        print(f"--- i:{i_big}, k:{k_big}")

        hx = 4 / i_big
        ht = time / i_big

        num.I_big = i_big
        num.K_big = k_big

        n = num.get_value_at(x, time)
        a = an.get_value_at(x, time)
        epsilon = abs(n - a)

        d = 0
        if i > 0:
            d = eps[i-1] / epsilon

        eps.append(epsilon)
        i_s.append(i_big)
        k_s.append(k_big)
        ht_s.append(ht)
        hx_s.append(hx)
        deltas.append(d)

        i_big *= 2
        k_big *= 4

    to_file('result.txt', i_s, k_s, ht_s, hx_s, eps, deltas)