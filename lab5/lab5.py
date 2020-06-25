import mpmath
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def f1(x):
    return mpmath.cos(x) * mpmath.cosh(x) - 1

def f2(x):
    return 1/x - mpmath.tan(x)

def f3(x):
    return mpmath.power(2, x) + mpmath.exp(x) + 2 * mpmath.cos(x) - 6

def f1_derivative(x):
    return mpmath.cos(x) * mpmath.sinh(x) + mpmath.sin(x) * mpmath.cosh(x)

def f2_derivative(x):
    return mpmath.sec(x) - 1/(mpmath.power(x, 2))

def f3_derivative(x):
    return  mpmath.exp(x) - mpmath.power(2, -x)*mpmath.ln(2) - 2 * mpmath.sin(x)

def bisection(precision, start, end, abs_error, function = f1):
    number_iteration = 1000000
    mpmath.mp.dps = precision
    u = function(mpmath.mpf(start))
    v = function(mpmath.mpf(end))
    e = end - start
    # print(start, end, u, v)
    if mpmath.sign(u) == mpmath.sign(v):
        print("sgn(function(start)) = sgn(function(end))")
        return
    else:
        for k in range(1, number_iteration + 1):
            e = e / 2
            c = start + e
            w = function(c)
            # print("Iteration = {0}, center partition c = {1}, f(c) = {2}, length of 1/2 partition = {3}".format(k, c, w, e))
            if abs(w) < abs_error:
                print("Iteration = {0}, center partition c = {1}, f(c) = {2}, length of 1/2 partition = {3}"
                      .format(k, c, w, e))
                return #c, w, k
            if mpmath.sign(w) != mpmath.sign(u):
                end = c
                v = w
            else:
                start = c
                u = w

def test_bisection():
    epsilon_arr = [10e-8, 10e-16, 10e-34]
    prec_arr = [10, 25, 60]
    for prec in prec_arr:
        mpmath.mp.dps = prec
        for epsilon in epsilon_arr:
            epsilon = mpmath.mpf(epsilon)
            print(f"{bcolors.HEADER}precision = {prec} \t epsilon = {epsilon} {bcolors.ENDC}")
            print(f"{bcolors.OKBLUE}function = cos(x)*cosh(x) - 1 {bcolors.ENDC}")
            bisection(prec, 3 * mpmath.pi / 2, 2 * mpmath.pi, epsilon, f1)
            print(f"{bcolors.OKBLUE}function = 1/x - tan(x) {bcolors.ENDC}")
            bisection(prec, 0 + epsilon, mpmath.pi / 2, epsilon, f2)
            print(f"{bcolors.OKBLUE}function = 2^x + e^x + 2*cos(x) {bcolors.ENDC}")
            bisection(prec, 1, 3, epsilon, f3)
            print("")


def newton_method(precision, x0, abs_error, function = f1, der_function =f1_derivative):
    number_iteration = 1000000
    mpmath.mp.dps = precision
    v = function(mpmath.mpf(x0))
    k = 0
    if abs(v) < abs_error:
        print("Iteration = {0}".format(k))
        return
    else:
        for k in range(1, number_iteration+1):
            x1 = x0 - v/(der_function(x0))
            v = function(x1)
            if abs(v) < abs_error:
                print("Iteration = {0}\t approximate value of the element = {1}".format(k, v))
                return
            else:
                x0 = x1

def test_newton_method():
    epsilon_arr = [10e-8, 10e-16, 10e-34]
    prec_arr = [10, 25, 60]
    for prec in prec_arr:
        mpmath.mp.dps = prec
        for epsilon in epsilon_arr:
            epsilon = mpmath.mpf(epsilon)
            print(f"{bcolors.HEADER}precision = {prec} \t epsilon = {epsilon} {bcolors.ENDC}")

            print(f"{bcolors.OKBLUE}function = cos(x)*cosh(x) - 1\tx0 = 3*pi/2 {bcolors.ENDC}")
            newton_method(prec, 3 * mpmath.pi / 2,  epsilon, f1)
            print(f"{bcolors.OKBLUE}function = cos(x)*cosh(x) - 1\tx0 = 2*pi {bcolors.ENDC}")
            newton_method(prec, 2 * mpmath.pi , epsilon, f1)

            print(f"{bcolors.OKBLUE}function = 1/x - tan(x) \t x0 = {epsilon} {bcolors.ENDC}")
            newton_method(prec, epsilon,epsilon, f2, f2_derivative)
            print(f"{bcolors.OKBLUE}function = 1/x - tan(x) \t x0 = pi/2 {bcolors.ENDC}")
            newton_method(prec, mpmath.pi / 2, epsilon, f2, f2_derivative)

            print(f"{bcolors.OKBLUE}function = 2^x + e^x + 2*cos(x)\t x0 = 1{bcolors.ENDC}")
            newton_method(prec, 1, epsilon, f3, f3_derivative)
            print(f"{bcolors.OKBLUE}function = 2^x + e^x + 2*cos(x)\t x0 = 3{bcolors.ENDC}")
            newton_method(prec, 3, epsilon, f3, f3_derivative)

            print("")

def secand_method(precision, start, end, abs_error, function = f1):
    number_iteration = 1000000
    mpmath.mp.dps = precision
    fstart = function(start)
    fend = function(end)
    for k in range(2, number_iteration+1):
        if abs(fstart) > abs(fend):
            start, end = end, start
            fstart, fend = fend, fstart
        if (fend -fstart) == 0:
            return
        else:
            s = (end - start)/(fend - fstart)
            end = start
            fend = fstart
            start = start - fstart * s
            fstart = function(start)
            if abs(fstart) < abs_error:
                print("Iteration = {0}\t approximate value of the element = {1}".format(k-1, fstart))
                return

def test_secand_method():
    epsilon_arr = [10e-8, 10e-16, 10e-34]
    prec_arr = [10, 25, 60]
    for prec in prec_arr:
        mpmath.mp.dps = prec
        for epsilon in epsilon_arr:
            epsilon = mpmath.mpf(epsilon)
            print(f"{bcolors.HEADER}precision = {prec} \t epsilon = {epsilon} {bcolors.ENDC}")

            print(f"{bcolors.OKBLUE}function = cos(x)*cosh(x) - 1 {bcolors.ENDC}")
            secand_method(prec, 3 * mpmath.pi / 2, 2 * mpmath.pi, epsilon, f1)
            print(f"{bcolors.OKBLUE}function = 1/x - tan(x) {bcolors.ENDC}")
            secand_method(prec, 0 + epsilon, mpmath.pi / 2, epsilon, f2)
            print(f"{bcolors.OKBLUE}function = 2^x + e^x + 2*cos(x) {bcolors.ENDC}")
            secand_method(prec, 1, 3, epsilon, f3)

            print("")


if __name__ == '__main__':
    PREC = 30
    # test_bisection()
    test_newton_method()
    # test_secand_method()