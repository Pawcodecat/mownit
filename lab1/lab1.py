from numpy import  float32
from numpy import  float64
from numpy import  linspace
import matplotlib.pyplot as plt
from datetime import datetime

#esercise1

x = float32(0.53125)
numbers = [x] * 10 ** 7
numbers2 = []
for i in range(0, 10 ** 4 ):
    if(i % 2 == 0):
        numbers2.append(10**10 * x)
    else:
        numbers2.append(x)



def sum():
    sum = float32(0)
    for i in range(0, 10 ** 7):
        sum += numbers[i]
    return  sum

def error():
    absolute_error = sum() - 5312500
    print(f"Sum of the next element: absolute_error = {absolute_error}")
    relative_error = abs(absolute_error / 5312500)
    print(f"Sum of the next element: relative_error  = {relative_error}")

def graph():
    sum = float32(0)
    print = []
    for i in range(0, 10 ** 7):
        sum += numbers[i]
        if i % 25000 == 0 and  i != 0:
            realtive_error = abs((sum - 0.53125 * (i+1))/ sum)
            print.append(realtive_error)

    plt.plot(print)
    plt.show()

def recursion_sum(arr, start, end):
    if start == end :
        return arr[start]
    else:
        mid = start + (end - start)//2
        return recursion_sum(arr, start, mid) + recursion_sum(arr,mid+1, end)

def error_recursion(arr,size,sum):
    start_time = datetime.now()
    absolute_error = recursion_sum(arr, 0, size -1) - sum
    end_time = datetime.now()
    print(f"Time of Recursion sum {end_time - start_time}")
    print(f"Recursion: absolute_error = {absolute_error}")
    realte_error = abs(absolute_error / sum)
    print(f"Recursion: relative_error= {realte_error}")

#exercise2

def kahan_algorithm(arr):

    sum = float32(0)
    err = float32(0)
    y = float32(0)
    temp = float32(0)
    for i in range(0, len(arr)):
        y = arr[i] - err
        temp = sum + y
        err = (temp - sum) - y
        sum = temp

    return sum

def blad_kahan_algorithms(arr, sum):
    start_time = datetime.now()
    absolute_error = kahan_algorithm(arr) - sum
    end_time = datetime.now()
    print(f"Time of Kahan algoritm {end_time - start_time}")
    print(f"Kahan algorithm absolute error = {absolute_error}")
    relative_error = abs(absolute_error / sum)
    print(f"kahan algorithm relative error = {relative_error}")

#exercise3

def dzeta_riemanna_forwrd_32(s, n):
    sum_32 = float32(0)
    k_32 = 1
    while(k_32 <= n):
        sum_32 += 1/(k_32**s)
        k_32 += 1
    return sum_32

def dzeta_riemanna_forwrd_64(s, n):
    sum_64 = float64(0)
    k_64 = 1
    while(k_64 <= n):
        sum_64 += 1/(k_64**s)
        k_64 += 1
    return sum_64

def dzeta_riemanna_backward_32(s, n):
    sum_32 = float32(0)
    k_32 = n
    while(k_32 >= 1):
        sum_32 += 1/(k_32**s)
        k_32 -= 1
    return sum_32

def dzeta_riemanna_backward_64(s, n):
    sum_64 = float64(0)
    k_64 = n
    while(k_64 >= 1):
        sum_64 += 1/(k_64**s)
        k_64 -= 1
    return sum_64


def eta_dirichleta_forwrd_32(s, n):
    sum_32 = float32(0)
    k_32 = 1
    while(k_32 <= n):
        sum_32 += ((-1)**k_32-1)/(k_32**s)
        k_32 += 1
    return sum_32

def eta_dirichleta_forwrd_64(s, n):
    sum_64 = float64(0)
    k_64 = 1
    while(k_64 <= n):
        sum_64 += ((-1)**k_64-1)/(k_64**s)
        k_64 += 1
    return sum_64

def eta_dirichleta_backward_32(s, n):
    sum_32 = float32(0)
    k_32 = n
    while(k_32 >= 1):
        sum_32 += ((-1)**k_32-1)/(k_32**s)
        k_32 -= 1
    return sum_32

def eta_dirichleta_backward_64(s, n):
    sum_64 = float64(0)
    k_64 = n
    while(k_64 >= 1):
        sum_64 += ((-1)**k_64-1)/(k_64**s)
        k_64 -= 1
    return sum_64


#exercise4

def logistic_mapping(r,xo, n):
    if n == 0:
        return xo
    return r*logistic_mapping(r, xo, n-1)*(1 - logistic_mapping(r, xo, n-1))

def logistic_equation(r, x):
    return r * x * (1-x)

def graph_logistic_mapping(x0, skip, iter, precision=float32, step=0.0001, r_min=1, r_max=4):
    rarr = []
    xnarr = []
    r_partition = linspace(r_min, r_max, int(1/step))

    for r in r_partition:
        xn = x0
        for i in range(iter + skip +1):
            if i >= skip:
                rarr.append(precision(r))
                xnarr.append(precision(xn))
            xn = precision(logistic_equation(r, xn));

    plt.plot(rarr, xnarr, ls='', marker='.', markersize=1)
    plt.ylim(0,1)
    plt.xlim(r_min, r_max)
    plt.xlabel('r')
    plt.ylabel('x')
    plt.title(f"x= {x0}, skip={skip}, iteration={iter}")
    plt.show()

def required_iterations(x0):
    r = float32(4)
    n = 0
    eps = 1e-6
    while x0 > eps and n < 10**5:
        x0= float32(logistic_equation(r, x0));
        n +=1
    print(f"required number of iteration to target value less than 10 ^-6 = {n}")




if __name__ == '__main__':

        #exercise1
        error()
        graph()
        error_recursion(numbers, 10**7, 5312500)
        error_recursion(numbers2, 10**4,2.65625*10**13)

        #exercise2
        blad_kahan_algorithms(numbers, 5312500)
        blad_kahan_algorithms(numbers2, 2.65625*10**13)

        #exercise3
        sarr = [2, 3.6667, 5, 7.2, 10]
        narr = [50, 100, 200, 500, 1000]
        counter = 0
        for s in sarr:
            for n in narr:
                print(f"dzeta_riemanna_forwrd_32( {s}, {n}) =  \t {dzeta_riemanna_forwrd_32(s, n)}")
                print(f"dzeta_riemanna_forwrd_64({s}, {n}) =   \t {dzeta_riemanna_forwrd_64(s, n)}")
                print(f"dzeta_riemanna_backward_32({s}, {n}) = \t {dzeta_riemanna_backward_32(s, n)}")
                print(f"dzeta_riemanna_forwrd_64({s}, {n}) =   \t {dzeta_riemanna_backward_64(s, n)}")
                counter += 4
                print(counter)

        counter = 0

        for s in sarr:
            for n in narr:
                print(f"eta_dirichleta_forwrd_32( {s}, {n}) =  \t {eta_dirichleta_forwrd_32(s, n)}")
                print(f"eta_dirichleta_forwrd_64({s}, {n}) =   \t {eta_dirichleta_forwrd_64(s, n)}")
                print(f"eta_dirichleta_backward_32({s}, {n}) = \t {eta_dirichleta_backward_32(s, n)}")
                print(f"eta_dirichleta_backward_64({s}, {n}) = \t {eta_dirichleta_backward_64(s, n)}")
                counter += 4
                print(counter)

        # exercise4
        print(logistic_mapping(3.8, 0.5, 10))
        plt.plot([1,2,3,4], [1, 4, 9, 19], 'o')
        plt.show()

        graph_logistic_mapping(0.25, 10, 10)
        graph_logistic_mapping(0.5, 10, 10)
        graph_logistic_mapping(0.75, 10, 10)

        graph_logistic_mapping(0.25, 10, 10, float32, 0.001, 3.75, 3.8)
        graph_logistic_mapping(0.5, 10, 10, float32, 0.001, 3.75, 3.8)
        graph_logistic_mapping(0.75, 10, 10, float32, 0.001, 3.75, 3.8)

        graph_logistic_mapping(0.25, 10, 10, float64, 0.001, 3.75, 3.8)
        graph_logistic_mapping(0.5, 10, 10, float64, 0.001, 3.75, 3.8)
        graph_logistic_mapping(0.75, 10, 10, float64, 0.001, 3.75, 3.8)

        required_iterations(0.25)
        required_iterations(0.5)
        required_iterations(0.75)




