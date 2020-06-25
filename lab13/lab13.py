import numpy as np
from matplotlib import pyplot as plt

class Point:
    x: int
    y: int
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_point(self, point):
        return np.sqrt((self.x -point.x)**2 + (self.y-point.y)**2)

def path_length(points):
    n = len(points)
    length = 0
    for i in range(n):
        length += points[i].distance_point(points[(i + 1) % n])
    return length

def generate_uniform_points(n, x_low, x_high, y_low, y_high):
    points_x = np.array([])
    points_y = np.array([])
    points = np.array([])

    points_x = np.append(points_x, np.random.uniform(low=x_low, high=x_high, size=n))
    points_y = np.append(points_y, np.random.uniform(low=y_low, high=y_high, size=n))

    for i in range(n):
        points = np.append(points, Point(points_x[i], points_y[i]))

    return points

def plot(points, energies_lenghts, best_enegries, iters):
    n = len(points)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    plt.figure(figsize=[30, 10])
    ax1.set(xlabel = "X")
    ax1.set(ylabel = "Y")
    ax1.set_title("Path with connected first and last point")
    ax1.plot( np.append([points[i].x for i in range(n)], points[0].x), np.append([points[i].y for i in range(n)], points[0].y),  "r")
    ax1.plot( [points[i].x for i in range(n)],[points[i].y for i in range(n)] ,  "bo")

    ax2.plot(np.arange(1, len(energies_lenghts) +1) ,energies_lenghts,  "b")
    ax2.set(xlabel = "Iteration")
    ax2.set(ylabel = "Length path with random swap")
    ax2.set_title("Length of paths")

    ax3.plot(iters, best_enegries, "b")
    ax3.set(xlabel = "Iteration")
    ax3.set(ylabel = "Length shortest path")
    ax3.set_title("Length of paths")
    plt.show()



import random

def move_rand_point_to_rand_pos(points):
    N = len(points)
    new_points = points[:]

    i = random.randint(1, N - 1)
    a = new_points[i]
    new_points = np.delete(new_points, i)

    j = random.randint(1, N - 2)
    new_points = np.insert(new_points, j, a)

    return new_points

def cut_swap_cut_reverse_first(points):
    n = len(points)

    i = random.randint(0, n / 2)
    c, d = np.split(points, [i])
    points = np.concatenate([d, c])

    i = random.randint(0, n / 2)
    c, d = np.split(points, [i])
    c = np.flip(c)
    new_points = np.concatenate([c, d])

    return new_points

def arbitrary_swap(points):
    n = len(points)

    i = random.randint(1, n - 1)
    j = random.randint(1, n - 1)
    points[i], points[j] = points[j], points[i]

    return points

#parametr beta is inverse temperature



# points = generate_uniform_points(20, -5, 5, -5, 5)
# simulated_annealing(arbitrary_swap, 10e4, points)


# simulated_annealing(10e4, 40)
#
# print(move_rand_point_to_rand_pos(np.array([0,1,2,3,4,5,6,7,8,9])))
# print(cut_swap_cut_reverse_first(np.array([0,1,2,3,4,5,6,7,8,9])))
# print(swap_two_points(np.array([0,1,2,3,4,5,6,7,8,9])))

def generate_gauss_points(n, a, b ,c, d):
    points = np.array([])
    points_x, points_y = np.random.multivariate_normal([0,0], [[a, b], [c, d]], n).T
    for i in range(n):
        points = np.append(points, Point(points_x[i], points_y[i]))

    return points


# plot(generate_gauss_points(20, 14, 77, 45, 5))

    # plt.plot(x, y, 'o')
    # plt.axis('equal')
    # plt.xlim((-40,40))
    # plt.ylim((-40, 40))
    # plt.title("a= %d b=%d c=%d d=%d" % (a,b,c,d))
    # print("a= %d b=%d c=%d d=%d" % (a,b,c,d))
    # plt.show()

def group_gauss(n):
    points = generate_gauss_points(n, 14, 77, 45, 5)
    points = np.append(points, generate_gauss_points(n, 25, 74, 87, 98))
    points = np.append(points, generate_gauss_points(n, 56, 19, 20, 28))
    points = np.append(points, generate_gauss_points(n, 70, 15, 0, 60))
    return points

def group_uniform_points(n):
    points = np.array([])
    for i in range(3):
        for j in range(3):
            points = np.append(points, generate_uniform_points(n, 20*i, 20*i + 10, 20*j, 20*j + 10))
    return points

# simulated_annealing(10e4, group_uniform_points(8))

# simulated_annealing(10e4, group_gauss(10))

def consecutive_swap(points):
    n = len(points)

    i = random.randint(1, n - 1)
    points[i], points[i+1] = points[i+1], points[i]

    return points


def swap_simulated_annealing_diff(func,points ,beta, n_accept, best_energy):
    n = len(points)
    energy = path_length(points)
    new_path = False

    if n_accept >= 100 * np.log(n):
        beta *=1.005

    new_points = func(points)

    new_energy = path_length(new_points)
    if random.uniform(0.0, 1.0) < np.exp(-beta * (new_energy - energy)):
        n_accept += 1
        energy = new_energy
        points = new_points[:]
        if energy < best_energy:
            best_energy = energy
            best_path = points[:]
            new_path = True

    return points, beta, n_accept, best_energy, new_path


def simulated_annealing_diff(func, no_of_iter, points):
    energy_min = float('inf')
    beta = 1.0
    n_accept = 0
    iteration = 0
    best_energies = []
    while iteration < no_of_iter:
        new_points, beta, n_accept, energy_min, new_path = \
            swap_simulated_annealing_diff(func, points[:], beta, n_accept, energy_min)
        if new_path:
            print("Path was shortened. No of iteration = %d\tNo of shortening path = %d\t Length of path = %4.2f"
                  % (iteration, n_accept, energy_min))
            plot(new_points)
            best_energies.append((iteration, energy_min))
        iteration += 1
    return best_energies


# points = generate_uniform_points(20, -5, 5, -5, 5)
# arbitrary_path_lengths = simulated_annealing_diff(arbitrary_swap, 10e3, points)


def consecutive_swap(points):
    n = len(points)

    i = random.randint(1, n - 1)
    points[i], points[i + 1] = points[i + 1], points[i]

    return points


# consecutive_path_lengths = simulated_annealing_diff(arbitrary_swap, 10e3, points)


def diff_arbitrary_consecutive():
    # tuple to two array
    arbitrary_x, arbitrary_y = zip(*arbitrary_path_lengths)
    plt.plot(arbitrary_x, arbitrary_y, label = "arbitrary swap")

    consecutive_x, consecutive_y = zip(*consecutive_path_lengths)
    plt.plot(consecutive_x, consecutive_y, label = "consecutive swap")

    plt.legend(loc="upper left")
    plt.ylabel('PLength')
    plt.xlabel('Iteration')
    plt.title("Comparison arbitrary swap and consecutive swap");
    plt.show()


# diff_arbitrary_consecutive()


def swap_simulated_annealing_temp(points ,beta, n_accept, best_energy):
    n = len(points)
    energy = path_length(points)
    new_path = False

    if n_accept >= 100 * np.log(n):
        beta *=1.005

    p = np.random.uniform(0.0, 1.0)
    if p < 0.2:
        new_points = cut_swap_cut_reverse_first(points)
    elif p < 0.6:
        new_points = move_rand_point_to_rand_pos(points)
    else:
        new_points = arbitrary_swap(points)

    new_energy = path_length(new_points)
    if random.uniform(0.0, 1.0) < np.exp(-beta * (new_energy - energy)):
        n_accept += 1
        energy = new_energy
        points = new_points[:]
        if energy < best_energy:
            best_energy = energy
            best_path = points[:]
            new_path = True

    return points, beta, n_accept, best_energy, new_path, energy


def simulated_annealing_temp(no_of_iter, points, temperature):
    energy_min = float('inf')
    beta = 1/temperature
    n_accept = 0
    iter = 0
    best_energies = []
    while iter < no_of_iter:
        iter += 1
        new_points, beta, n_accept, energy_min, new_path, old_energy = \
            swap_simulated_annealing_temp(points[:], beta, n_accept, energy_min)
        if new_path:
            best_energies.append((iter,  energy_min))

    return best_energies

def diff_temperature():
    for temp in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        temp_paths = simulated_annealing_temp(10e3, generate_uniform_points(20, -5, 5, -5, 5), temp)
        x, y = zip(*temp_paths)
        plt.plot(x, y, label="temp = {}".format(temp))



    plt.legend(loc="upper right")
    plt.ylabel('Length paths')
    plt.xlabel('Iteration')
    plt.title("Comparison hogh and low temperature");
    plt.show()


diff_temperature()