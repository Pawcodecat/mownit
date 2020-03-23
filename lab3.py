import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg

def sphere():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    s = np.linspace(0, 2 * np.pi, 100)
    t = np.linspace(0, np.pi, 100)

    x =  100*np.outer(np.cos(s), np.sin(t))
    y =  100*np.outer(np.sin(s), np.sin(t))
    z =  100*np.outer(np.ones(np.size(s)), np.cos(t))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='g')


    # a  = [1,1,1]
    # A1 = [[100,2,30],[2,45,143],[1,3,45]]
    # A1 = np.array(A1)
    # A2 = [[100,2,30],[2,45,143],[1,3,45]]
    # A2 = np.array(A2)
    # A3 = [[100,2,30],[2,45,143],[1,3,45]]
    # A3 = np.array(A3)
    # B1 = a @ A1
    # B2 = a @ A2
    # B3 = a @ A3

    x = np.outer(B1[0] * np.cos(s), np.sin(t))
    y = np.outer(B1[1] * np.sin(s), np.sin(t))
    z = np.outer(B1[2] * np.ones(np.size(s)), np.cos(t))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
    plt.show()

    print(B1)

#print(a)
# U, s, Vh = linalg.svd(a)
# print(U)
# print()
# print(s)
# print()
# print(Vh)

if __name__ == '__main__':
    sphere();