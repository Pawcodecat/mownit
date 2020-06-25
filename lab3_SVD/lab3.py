import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg
import random
import sys
import os
from PIL import Image

def make_X_Y_Z_elipse(A):
    s = np.linspace(0, 2 * np.pi, 100)
    t = np.linspace(0, np.pi, 100)

    x = np.outer(A[0][0] * np.cos(s), np.sin(t))  +\
        np.outer(A[0][1] * np.sin(s), np.sin(t)) +\
        np.outer(A[0][2] * np.ones(np.size(s)), np.cos(t))

    y = np.outer(A[1][0] * np.cos(s), np.sin(t))  + \
        np.outer(A[1][1] * np.sin(s), np.sin(t))  + \
        np.outer(A[1][2] * np.ones(np.size(s)), np.cos(t))

    z = np.outer(A[2][0] * np.cos(s), np.sin(t))  + \
        np.outer(A[2][1] * np.sin(s), np.sin(t)) + \
        np.outer(A[2][2] * np.ones(np.size(s)), np.cos(t))

    return x, y, z

def make_X_Y_Z_sphere():
    s = np.linspace(0, 2 * np.pi, 100)
    t = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(s), np.sin(t))
    y = np.outer(np.sin(s), np.sin(t))
    z = np.outer(np.ones(np.size(s)), np.cos(t))
    return x, y, z



def draw_3d(x, y, z, viev_elev=20, view_azim=30):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=viev_elev, azim=view_azim)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', alpha = 0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def draw_3d_with_singular_values(x, y, z, A):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', alpha=0.3)

    #draw singular values
    U, S, VT = np.linalg.svd(A)
    diag_S = np.diag(S)

    for row in diag_S:
        vector = np.dot(U, row)
        ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], linewidth=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def draw_sphere():
    x, y, z = make_X_Y_Z_sphere()
    draw_3d(x, y, z)


def draw_elipse(A):
    x, y, z = make_X_Y_Z_elipse(A)
    draw_3d(x, y, z)

def draw_elipse_with_singular_values(A):
    x, y, z = make_X_Y_Z_elipse(A)
    draw_3d_with_singular_values(x, y, z, A)

def transform_vectors():
    A1 = [[1, 0.02, 0.3], [0.02, 0.45, 1.43], [0.01, 0.03, 0.45]]
    A2 = [[0.5, -3.2, 0.3], [1.4, 1.2, -0.40], [0.7, -1.8, 0.65]]
    A3 = [[-2.3, 0.15, -2], [0.12, -0.45, 1], [1, -0.6, -1.2]]
    return A1, A2, A3

def draw_3_elipse():
    A1, A2, A3 = transform_vectors()
    draw_elipse(A1)
    draw_elipse(A2)
    draw_elipse(A3)


def draw_3_elipse_with_singular_values():
    A1, A2, A3 = transform_vectors()
    draw_elipse_with_singular_values(A1)
    draw_elipse_with_singular_values(A2)
    draw_elipse_with_singular_values(A3)

def find_100_proportion():
    A = [[1 for _ in range(3)] for _ in range(3)]
    S = [1] * 3
    while S[0] < 100 * S[2]:
        for i in range(3):
            for j in range(3):
                A[i][j] = random.uniform(-3, 3)

        U, S, VT = np.linalg.svd(A)
    diag_S = np.diag(S)
    print(A)
    print(diag_S)
    return A


def draw_100_proportion_elipse():
    A = find_100_proportion()
    x, y, z = make_X_Y_Z_elipse(A)
    draw_3d(x, y, z, 10, 90)
    draw_3d(x, y, z, 20, 40)
    draw_3d(x, y, z, 50, 80)
    draw_3d(x, y, z, 60, 40)
    draw_3d(x, y, z, 100, 100)

def show_transformation(A):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', alpha=0.3)

    # draw singular values
    U, S, VT = np.linalg.svd(A)
    diag_S = np.diag(S)

    for row in diag_S:
        vector = np.dot(U, row)
        ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], linewidth=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    for row in diag_S:
        vector = np.dot(U, row)
        ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], linewidth=5)

def show_transformation():
    A1, A2, A3 =transform_vectors()
    U, S, VT = np.linalg.svd(A2)
    diag_S = np.diag(S)
    draw_elipse_with_singular_values(VT)
    draw_elipse_with_singular_values(diag_S @ VT)
    draw_elipse_with_singular_values(U @ diag_S @ VT)



def lena_compression(k):
    path, extension = os.path.splitext("lena.png")
    img = Image.open("lena.png")
    width = img.width
    height = img.height
    img_arr = np.array(img)

    arr1 = img_arr.transpose(0, 2, 1).reshape(height * 3, width)
    arr2 = img_arr.transpose(1, 2, 0).reshape(width * 3, height)

    U, S, A1 = linalg.svd(arr1)
    U, S, A2 = linalg.svd(arr2)

    B1 = A1[:k*3, :]
    B2 = A2[:k*3, :]

    C1 = B1.T.dot(B1)
    C2 = B2.T.dot(B2)

    arr2 = np.tensordot(img_arr, C1, (1, 0))
    arr3 = np.tensordot(arr2, C2, (0, 0))
    arr4 = arr3.transpose(2, 1, 0)

    imgk = Image.fromarray(np.uint8(arr4))
    file = path + str(k) + extension
    imgk.save(file)

def compress_lena_arr():
    arr = [1, 3, 5, 10, 15, 50, 100, 200]
    for ele in arr:
        lena_compression(ele)

if __name__ == '__main__':
    # draw_sphere()
    # draw_3_elipse()
    # draw_3_elipse_with_singular_values()
    # draw_100_proportion_elipse()
    # show_transformation()
    compress_lena_arr()
