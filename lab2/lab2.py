import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate

#exercise1

def function(x):
    x =np.float32(x)
    return np.float32(1/(1+x**2))




def vandermond_matrix(start, end, n):
    d = np.linspace(start, end, n + 1)
    Y = []
    d = np.array(d)
    vander = []
    original_x = np.linspace(-5, 5, 1000)
    original_y = []

    #make vandermonde'a matrix the same
    # vander = np.vander(d, n+1, increasing=True)
    for point in d:
        x = []
        for i in range(n+1):
                x.append(np.float32(point**i))
        vander.append(x)

    #make matrix of value in n+1 points
    for i in d:
        Y.append(np.float32(function(i)))

    A = np.linalg.solve(vander, Y)

    result = []
    delta = []
    for point_x in original_x:
        sum = 0
        for (power, pol_factor) in enumerate(A):
            sum += pol_factor*point_x**power
        func_value = np.float32(function(point_x))
        result.append(sum)
        original_y.append(func_value)
        diff = func_value - sum
        delta.append(diff)

    plt.plot(original_x, original_y, 'b', markersize=1)
    plt.plot(original_x, result, "g", markersize=1)
    plt.plot(original_x, delta, "r", markersize=1)

    plt.show()

def vandermond_matrix_czebyszew(start, end, n):

    czybyszew = []
    for k in range(1, n + 1):
        point_x = 0.5 * (start + end) + 0.5 * (end - start) * math.cos(((2 * k - 1) / (2 * n)) * math.pi)
        czybyszew.append(point_x)

    y = []
    x = []
    for (i, c) in enumerate(czybyszew):
        x.append(c)
        y.append(function(c))

    x = np.array(x)
    y = np.array(y)

    van = np.vander(x, n, increasing=True)
    A = np.linalg.solve(van, y)
    result = []
    original_y = []
    delta = []
    original_x = np.linspace(-5, 5, 1000)


    for point_x in original_x:
        sum = 0
        for (power, pol_factor) in enumerate(A):
            sum += pol_factor*point_x**power
        func_value = np.float32(function(point_x))
        result.append(sum)
        original_y.append(func_value)
        diff = func_value - sum
        delta.append(diff)

    plt.plot(original_x, original_y, 'bo', markersize=1)
    plt.plot(original_x, result, 'go', markersize=1)
    plt.plot(original_x, delta, 'ro', markersize=1)
    plt.show()

def x_func(t):
    a = -3
    return a*math.cos(t)

def y_func(t):
    b = 4
    return b*math.sin(t)

def cubic_spline_ellipse():
    t_determine = np.linspace(0, 2*math.pi, 10)

    spline_x = interpolate.interp1d(t_determine, list(map(x_func, t_determine)), kind = 'cubic')
    spline_y = interpolate.interp1d(t_determine, list(map(y_func, t_determine)), kind = 'cubic')
    t_plot = np.linspace(0, 2*math.pi, 30)
    x_plot = []
    y_plot = []

    for t in t_plot:
        x_plot.append(spline_x(t))
        y_plot.append(spline_y(t))

    plt.plot(x_plot, y_plot)
    plt.show()





#if __name__ == 'main':


vandermond_matrix(-5, 5, 10)
vandermond_matrix(-5, 5, 15)
vandermond_matrix(-5, 5, 5)
vandermond_matrix_czebyszew(-5, 5, 15)
cubic_spline_ellipse()


