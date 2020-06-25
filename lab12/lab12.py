
def derivative_dydx(x, y):
    return ((x - y) / 2)



def runge_kutta(a, alfa, b, N):

    h = ((b - a) / N)
    t = a
    w = alfa
    for i in range(1, N+1):

        k1 = h * derivative_dydx(t, w)
        k2 = h * derivative_dydx(t + 0.5 * h, w + 0.5 * k1)
        k3 = h * derivative_dydx(t + 0.5 * h, w + 0.5 * k2)
        k4 = h * derivative_dydx(t + h, w + k3)

        w += (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        t = a + i * h
    return w

x0 = 0
y = 1
x = 2
h = 10
print ('The value of y at x is:', runge_kutta(x0, y, x, h))

# print(sp.integrate.RK45((x-y)/2, 0, 1, 2))

from math import sqrt

import numpy as np

def runge_kutta_oroder_4(f, a, alfa, b, n):
    vx = np.zeros(n+1)
    vy = np.zeros(n+1)
    h = (b - a) / float(n)
    vx[0] = x = a
    vy[0] = w = alfa
    for i in range(1, n + 1):
        k1 = h * f(x, w)
        k2 = h * f(x + 0.5 * h, w + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, w + 0.5 * k2)
        k4 = h * f(x + h, w + k3)
        vx[i] = x = a + i * h
        vy[i] = w = w + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vx, vy

from sympy import *

import pandas as pd
def f3(x, t):
    return (100*(sin(t)-x))


def test_func():
    for no_of_parts in 200, 150, 120, 100:
        vx, vy = runge_kutta_oroder_4(f3, 0, 0, 3, no_of_parts)
        print("h = %1.3f" % (float(3/no_of_parts)))
        for x, y in list(zip(vx, vy))[::20]:
            print("%4.2f %10.4f" % (x, y))
        print("")


def runge_kutta_fehlberg(f, a, b, x0, tol, h_max, h_min):
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5

    t = b
    x = x0
    h = h_max

    T = np.array([t])
    X = np.array([x])

    while t > a:
        # Adjust last interval
        if t - h < a:
            h = t - a

        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)

        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
        #         if len( np.shape( r ) ) > 0:
        #             r = max( r )
        if r <= tol:
            t = t - h
            x = x - (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5)
            T = np.append(T, t)
            X = np.append(X, [x], 0)

        r = r if r > 0 else tol
        h *= min(max(0.84 * (tol / r) ** 0.25, 0.1), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            print("Error: minimum h exceeded. Procedure completed unsuccessfully. hmin = %1.2e" % (h_min) )
            break;

    return (T, X)


def f3(x, t):
    return (3*x/t + 9*t/2 -13)

def test_func():
    vx, vy = runge_kutta_fehlberg(f3, 0.5, 3, 6, 10e-9, 10e-3, 10e-10)
    print("t\t\tfunction x\t x")
    for x, y in list(zip(vx, vy))[::20]:
        print("%4.2f %10.4f %10.4f" % (x, y, x**3 - 9/2 * x**2  + 13 * x /2))
    x = vx[len(vx) - 1]
    y = vy[len(vy) - 1]
    print("%4.2f %10.4f %10.4f" % (x, y, x**3 - 9/2 * x**2  + 13 * x /2))

test_func()

