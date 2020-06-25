import numpy as np
from datetime import datetime
import random
from matplotlib import pyplot as plt
import networkx as nx

from scipy.linalg import lu_factor, lu_solve, lu
def make_matrix(N=600, max=10000):
    return max * np.random.random_sample((N,N))
def make_B(N=600, max=10000):
    return max * np.random.random_sample(N)
def make_list(N=600, max=10000):
    return [[random.uniform(1, max) for _ in range(N)] for _ in range(N)]



def Gauss_Jordan_method(A, N=600):
    p = []
    s = []
    for i in range(N):
        p.append(i)
        s.append(np.amax(A[i]))

    for k in range(N-1):
        max = abs(A[p[k]][k]) / s[p[k]]
        j = k
        for i in range(k+1,N):
            if abs(A[p[i]][k]) / s[p[i]] > max:
                max = abs(A[p[i]][k]) / s[p[i]]
                j = i
        p[k], p[j] = p[j], p[k]
        for i in range(k+1, N):
            z = A[p[i]][k] / A[p[k]][k]
            A[p[i]][k] = z
            for j in range(k+1, N):
                A[p[i]][j] = A[p[i]][j] - (z * A[p[k]][j])

    return A, p



def solve_Ax_b_using_PA_LU(A,p,B, N=600):
    X = [0]*N
    for k in range(0, N-1):
        for i in range(k+1, N):
            B[p[i]] = B[p[i]] - (A[p[i]][k] * B[p[k]])

    for i in range(N-1, -1, -1):
        bpi = B[p[i]]
        for j in range(i+1, N):
            bpi  = bpi - (A[p[i]][j] * X[j])
        X[i] = bpi / A[p[i]][i]

    return X

def compare_elimination_Gauss_library():
    print("Function\ttime")
    for i in range(3):
        A = make_matrix()
        B = make_B()

        startTime = datetime.now()
        A, p = Gauss_Jordan_method(A)
        X = solve_Ax_b_using_PA_LU(A, p, B)
        endTime = datetime.now()
        print("MY      \t{0}".format(endTime-startTime))

        startTime = datetime.now()
        np.linalg.solve(A, B)
        endTime = datetime.now()
        print("LIB     \t{0}".format(endTime-startTime))

def extract_L_U_from_Gauss(A, N=600):
    L = [[0 for _ in range(N)] for _ in range(N)]
    U = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        L[i][i] = 1
        U[i][i] = A[i][i]
    for i in range(1, N):
        for j in range(0, i):
            L[i][j] = A[i][j]
            U[N-i-1][N-j-1] = A[N-i-1][N-j-1]
    return L, U


def LU_factorization(A):
    A = np.copy(A)
    N = len(A[0])
    p = np.arange(0, N)
    for k in range(N - 1):
        max = A[k][k]
        j = k
        for i in range(k+1 , N):
            if abs(A[i][k]) > max:
                max = abs(A[i][k])
                j = i

        p[k], p[j] = p[j], p[k]
        A[j], A[k] = A[k], A[j]

        for i in range(k + 1, N):
            A[i, k] /=  A[k, k]
            for j in range(k + 1, N):
                A[i, j] -= A[i, k] * A[k, j]


    L, U = extract_L_U_from_Gauss(A, N)

    return L, U






def printMatrix(s, name):

    # Do heading
    print("{0}    ".format(name), end="")
    for j in range(len(s[0])):
        print("%5d " % j, end="")
    print()
    print("     ", end="")
    for j in range(len(s[0])):
        print("------", end="")
    print()
    # Matrix contents
    for i in range(len(s)):
        print("%3d |" % (i), end="") # Row nums
        for j in range(len(s[0])):
            print("%5.2f " % (s[i][j]), end="")
        print()
    print()

def printMatrix_numpy(s, name):

    # Do heading
    print("{0}    ".format(name), end="")
    for j in range(s[0].shape):
        print("%5d " % j, end="")
    print()
    print("     ", end="")
    for j in range(s[0].shape):
        print("------", end="")
    print()
    # Matrix contents
    for i in range(s[0].shape):
        print("%3d |" % (i), end="") # Row nums
        for j in range(s[0].shape):
            print("%5.2f " % (s[i][j]), end="")
        print()
    print()

def compare_factorization():
    A = [[4.0, 1.0, 3.0],
         [1.0, 4.0, 1.0],
         [2.0, 3.0, 2.0]]


    np.set_printoptions(suppress=True)

    N = 4



    L, U = LU_factorization(A)
    L = np.array(L)
    U = np.array(U)

    print("MY ")
    print("L")
    print(L)
    print("U")
    print(U)
    print("L*U")
    print(np.dot(L, U))




def show_graph(G,scheme, name):
    plt.figure(name)

    nx.draw_networkx_nodes(G, scheme, node_color="red", node_size=1000 )
    nx.draw_networkx_edges(G, scheme, width = 2, alpha=1)
    nx.draw_networkx_labels(G, scheme, font_size=14, font_color="white")
    nx.draw_networkx_edge_labels(G, scheme, edge_labels=)








if __name__ == '__main__':
    # A = [[4, 1, 3, 2],
    #      [1, 4, 1, 5],
    #      [2, 3, 2, 3],
    #      [4, 5, 3, 8]]
    # B =make_list(4,20)
    # compare_factorization()
    # compare_elimination_Gauss_library()
    # L, U = LU_factorization(B)
    # print(np.dot(L, U))
