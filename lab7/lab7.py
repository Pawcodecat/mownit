import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time

import matplotlib.dates as mdates

def generate_matrix(N):
    return np.random.rand(N,N)


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A, max_iteration, eps):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    for i in range(max_iteration):
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < eps:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

def check_power_iteration():
    np.set_printoptions(suppress=True)
    j = 0
    for i in range(4, 10):
        j += 1
        print("TEST NR {0}".format(j))
        arr = generate_matrix(i)
        copy_arr = np.copy(arr)
        e_val, e_vec = power_iteration(arr, 10000, 0.01)
        print("my power iteration")
        print(e_val)
        print(e_vec)
        e_val, e_vec = np.linalg.eig(copy_arr)
        print("library power iteration")
        print(e_val)
        print(e_vec)

def test_power_iteration():
    my_func = []
    lib_func = []
    sizes = []
    for size in range (100,500,20):
        sizes.append(size)
        arr = generate_matrix(size)
        copy_arr = np.copy(arr)

        start_time = time.time()
        np.linalg.eig(copy_arr)
        diff_time = time.time() - start_time
        lib_func.append(diff_time * 1000)

        start_time = time.time()
        power_iteration(arr,1000000,0.1)
        diff_time = time.time() - start_time
        my_func.append(diff_time*1000)




    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax1.set_xlabel('N')
    ax1.set_ylabel('miliseconds')
    plt.plot(sizes, my_func, label="my function")
    plt.plot(sizes, lib_func, label="library fuction")
    plt.legend(loc="upper left")

    plt.show()



def inverse_power_iteration(sigma, A):
    n, d = A.shape
    x0 = np.array((np.ones(n)))
    P,  L, U= scipy.linalg.lu(A-sigma*np.eye(n))
    v = x0
    for i  in range(n):
        v = np.lusolve(L, U, P, v)
        v = v/np.norm(v);
    return v

def inverse_power_method(A, sgn, eps, max_iterations):
    size, d = A.shape
    x0 = np.array([1.5 for i in range(size)])
    x0 = np.array(x0/np.linalg.norm(x0, ord=np.inf))

    for i in range(size):
        A[i][i] = A[i][i] - sgn

    flag = False
    LU = scipy.linalg.lu_factor(A)

    for i in range(max_iterations):
        x1 = scipy.linalg.lu_solve(LU, x0)
        x2 = x1/np.linalg.norm(x1, ord=np.inf)

        if np.linalg.norm(x2 - x0) < eps:
            flag = True

        x0 = x2
        if flag:
            break

    return x1/np.linalg.norm(x1)

def check_inverse_power_method():
    A = generate_matrix(3)

    e_val, e_vec = scipy.linalg.eig(A)
    print("Library fuction:")
    print(e_val)
    print(e_vec)
    print("My function:")
    print("Sigma equal approximately ", e_val[0])
    print(inverse_power_method(A, e_val[0] + 0.1, 0.01, 10000))
    print("Sigma equal approximately  ", e_val[1])
    print(inverse_power_method(A, e_val[1] + 0.2, 0.01, 10000))
    print("Sigma equal approximately  ", e_val[2])
    print(inverse_power_method(A, e_val[2] + 0.5, 0.01, 10000))


if __name__ =="__main__":
    # check_power_iteration()
    #test_power_iteration()
   # print(eigenvalue(np.array([[-2, -3], [6, 7]]), np.array([1,1])))
   # print(inverse_power_iteration(1.1, np.array([[-2, -3], [6, 7]])))
   check_inverse_power_method()
