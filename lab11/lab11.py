import numpy as np
import random

from numpy.random import default_rng
from matplotlib import pyplot as plt


def plot(x, generator):

    plt.hist(x, density=False, bins=10, label="Data")

    plt.legend(loc="upper left")
    plt.ylabel('Number of random digits')
    plt.xlabel('Data')
    plt.title("Histogram {0} All digits={1}".format(generator, len(x)))
    plt.show()


# def distribution_10_partitions():
#     for n in [10, 1000, 5000]:
#         pcg64 = np.zeros(10)
#         mersene_twister = np.zeros(10)
#
#         for _ in range(n):
#             pcg64_number = default_rng().random()
#             pcg64[int((pcg64_number * 10) // 1)] += 1
#
#             mersene_twister_number = default_rng().random()
#             mersene_twister[int((mersene_twister_number * 10) // 1)] += 1


def distribution_10_partitions():
    for n in [10, 1000, 5000]:
        pcg64 = []
        mersene_twister = []

        for _ in range(n):

            pcg64.append(default_rng().random())
            mersene_twister.append(random.random())
        plot(pcg64, "PCG64")
        plot(mersene_twister, "Mersenne Twister")

def check_inequality_between_previous_and_next():
    for n in [10, 1000, 5000]:
        pcg64 = 0
        mersenne_twister = 0
        for i in range(n):
            next_pcg64 = default_rng().random()
            next_meresenne_twister = random.random()
            if i > 0 and next_pcg64 > previous_pcg64:
                pcg64 += 1
            if i > 0 and next_meresenne_twister > previous_mersenne_twister:
                mersenne_twister += 1

            previous_pcg64 = next_pcg64
            previous_mersenne_twister = next_meresenne_twister
        print( "GENERATOR: PCG64\t\t\t Number of digits: {0}\t Number of digits satisfy inequality: {1}"
               .format(n,pcg64))
        print( "GENERATOR: Mersenne twister\t Number of digits: {0}\t Number of digits satisfy inequality: {1}"
               .format(n,mersenne_twister))





# print(random.random())

from scipy import special as spc
def float_bin(number):
    res = ''
    for i in range(100):
        number *= 2
        res += str(int((number//1)))
        if number > 1:
            number %= 1
        elif number == 1:
            number = 0
            break
        # print(number)
    return res

def monobit(bin_data: str):
    count = 0
    # If the char is 0 minus 1, else add 1
    for char in bin_data:
        if char == '0':
            count -= 1
        else:
            count += 1
    # Calculate the p value
    sobs = count / np.sqrt(len(bin_data))
    p_val = spc.erfc(np.fabs(sobs) / np.sqrt(2))
    return p_val

def monobit_frequency_test():
    for n in [10, 1000, 5000]:
       non_random_pcg64 = 0
       non_random_mersenne_twister = 0
       for _ in range(n):
            pcg64 = default_rng().random()
            meresenne_twister = random.random()
            if monobit(float_bin(pcg64)) < 0.01:
                non_random_pcg64 += 1
            if monobit(float_bin(meresenne_twister)) < 0.01:
                non_random_mersenne_twister += 1

       print("GENERATOR: PCG64\t\t\t Number of digits: {0}\t Number of digits which are no-random: {1}"
                .format(n, non_random_pcg64))
       print("GENERATOR: mersenne_twister\t\t Number of digits: {0}\t Number of digits which are no-random: {1}"
                .format(n, non_random_mersenne_twister))



# digit = default_rng().random()
# print(digit)
# print(float_bin(digit))
# monobit_frequency_test()

def gaussian(u1,u2, mu, sigma):
  z1 = sigma * np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2) + mu
  z2 = sigma * np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2) + mu
  return z1, z2


# def box_muller_generator(mu = 0.0, sigma = 1.0):
#     for n in [10, 100, 5000]:
#         u1 = np.random.uniform()
#         u2 = np.random.uniform()
#         print(u1)
#         print(u2)
#         z1, z2 = gaussian(u1, u2, mu, sigma)
#         plt.figure()
#         plt.subplot(221)  # the first row of graphs
#         plt.hist(u1)  # contains the histograms of u1 and u2
#         plt.subplot(222)
#         plt.hist(u2)
#         # plt.subplot(223)  # the second contains
#         # plt.hist(z1)  # the histograms of z1 and z2
#         # plt.subplot(224)
#         # plt.hist(z2)
#         plt.show()


# box_muller_generator()


import scipy.stats as st

def pnorm(mu = 0, sigma = 1):
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    r = np.sqrt(-2*np.log(u1))
    a = 2*np.pi*u2
    return sigma*r*np.cos(a) + mu, sigma*r*np.sin(a) + mu

def box_muller_generator(mu = 0, sigma = 1):
    for n in [10, 100, 5000]:
        x = []
        for _ in range(n): x += pnorm(mu, sigma)
        plt.hist(x, density=True, bins=10, label="Data")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, n)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title("Histogram");
        plt.show()

def shapiro_wilk_test_20(mu = 0, sigma = 1):
        teretical_value = 0.905
        n = 10
        x = []
        for _ in range(n): x += pnorm(mu, sigma)
        x.sort()
        avg = sum(x) / len(x)
        S_2 = sum((i-avg)**2 for i in x)
        weights_a = [0.4734, 0.3211, 0.2565, 0.2085, 0.1686, 0.1334, 0.1013, 0.0711, 0.0422, 0.0140]
        b_2 = (sum((x[19-i] - x[i])*weights_a[i] for i in range(10)))**2
        W = b_2 / S_2
        print(W)
        if W > teretical_value:
            print("Hypothesis null accepted")
        else:
            print("Hypothesis null rejected")


def shapiro_wilk_test(mu = 0, sigma = 1):
    alfa = 0.05
    for n in [100, 5000]:
        x = []
        for _ in range(n): x += pnorm(mu, sigma)
        W, p_value = st.shapiro(x[0:n])
        print("p value: {0}".format(p_value))
        if p_value > alfa:
            print("Hypothesis null accepted")
        else:
            print("Hypothesis null rejected")

def plot_absolute_error(absolute_errors, points_one_iter, no_of_intervals):
    no_of_points = np.linspace(points_one_iter, no_of_intervals*points_one_iter, no_of_intervals)
    plt.subplot(1, 2, 2)
    plt.plot(no_of_points, absolute_errors, label="absolute errors")
    plt.legend(loc="upper left")
    plt.ylabel('Absolute error')
    plt.xlabel('Number of points')
    plt.title("Relationship between no of points and abs error")
    plt.show()

import time

def approximate_pi_monte_carlo(N, no_of_intervals = 30, time_interval = 30):
    x = np.array([])
    y = np.array([])

    absolute_errors = np.array([])
    points_one_iteration = int(N/no_of_intervals)

    for i in range(no_of_intervals):
        plt.figure(figsize=[14, 7])
        x = np.append(x, np.random.uniform(low=-1, high=1, size=[points_one_iteration, 1]))
        y = np.append(y, np.random.uniform(low=-1, high=1, size=[points_one_iteration, 1]))
        if_inside = x ** 2 + y ** 2 < 1

        approx_pi = 4 * np.sum(if_inside) / ((i+1)*points_one_iteration)
        absolute_errors = np.append(absolute_errors, abs(np.pi - approx_pi))

        print('Pi : {}, approximation: {}'.format(np.pi, approx_pi))

        x_in = x[if_inside]
        y_in = y[if_inside]
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, s=1)
        plt.scatter(x_in, y_in,color='r', s=1)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title("Total number of points = {}".format((i+1)*points_one_iteration))
        time.sleep(time_interval*10e-3)

        plot_absolute_error(absolute_errors, points_one_iteration, i+1)
        plt.show()


approximate_pi_monte_carlo(20000)

