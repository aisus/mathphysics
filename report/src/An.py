from sympy import *


def a_m(m: int):
    x, z = symbols('x z')
    l1 = legendre_poly(m, x)
    l2 = l1.subs(x, z)
    expr = l2 * (z ** 4)
    integral = integrate(expr, (z, -1, 1))
    return (integral * (2 * m + 1)) / 2


if __name__ == "__main__":
    init_printing()
    for i in range(120):
        print(f"A_{i} = {latex(a_m(i))}")
