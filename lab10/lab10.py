import numpy as np


A = np.array([0, 1, 2, 3, 3, 2, 1,0])

PI_2 = np.pi * 2.0

def dft(fn):
    with np.printoptions(precision=3, suppress=True, formatter={'float': '{:0.1e}'.format}, linewidth=200):
        N = len(fn)
        dft_vector = np.zeros(N, dtype = complex)
        for j in range(N):
            for k in range(N):
                dft_vector[j] += fn[k] * np.exp(- 1j * PI_2 * j * k / N)

        return dft_vector



def idft(fn):
    with np.printoptions(precision=3, suppress=True, formatter={'float': '{:0.1e}'.format}, linewidth=200):
        N = len(fn)
        signal_vector = np.zeros(N, dtype = complex)
        for j in range(N):
            for k in range(N):
                signal_vector[j] += fn[k] * np.exp(1j * PI_2 * j * k / N)

        return signal_vector/N

from matplotlib import pyplot as plt






def test_dft():
    fs = 1e2  # Sampling frequency
    dt = 1 / fs  # Sampling interval

    t = np.arange(0, 1, dt)

    # Generate white noise signal
    x = np.random.randn(*t.shape)
    X = dft(x)
    X_ref = np.fft.fft(x)
    X_err = np.abs(X - X_ref) ** 2

    # RMSE(Root Mean Squared Error)
    X_RMSE = np.sqrt(np.mean(X_err))

    print('RMSE:', np.round(X_RMSE, 15))
    print('TEST: succeeded' if X_RMSE < 1e-10 else 'TEST : failed RMSE too big')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    ax.plot(X_err)
    ax.set_ylabel('Frequency bin error')
    ax.set_xlabel('Frequency bin')
    plt.grid()
    plt.show()


def  test_idft():
    fs = 1e2  # Sampling frequency
    dt = 1 / fs  # Sampling interval

    t = np.arange(0, 1, dt)
    x = np.random.randn(*t.shape)
    X_err = np.abs(x - idft(dft(x))) ** 2

    # RMSE
    X_RMSE = np.sqrt(np.mean(X_err))

    print('RMSE:', np.round(X_RMSE, 15))
    print('TEST: succeeded' if X_RMSE < 1e-10 else 'TEST : failed RMSE too big')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    ax.plot(X_err)
    ax.set_ylabel('Frequency bin error')
    ax.set_xlabel('Frequency bin')
    plt.grid()
    plt.show()

def fft(fn):
    N = len(fn)
    if N == 1:
        return fn
    omega_n = np.exp(PI_2 * 1j/N)
    omega = 1
    a_0 =fn [::2]
    a_1 =fn [1::2]
    y_0 = fft(a_0)
    y_1 = fft(a_1)
    y = np.zeros(N, dtype = complex)
    for k in range(N//2):
        y[k] = y_0[k] + omega * y_1[k]
        y[k + N//2] = y_0[k] - omega * y_1[k]
        omega *= omega_n
    return y


from tqdm.auto import tqdm
import time

def test_complexity_fft_dft():
    # Number of repetitions for each test point on characteristic
    R = 100

    DFT_IMPLEMENTATIONS_LIST = {
        'dft': (dft, 7),
        'USER fft': (fft, 13),
        'LIB np.fft.fft': (np.fft.fft, 20),
    }

    results = {
        'dft': None,
        'USER fft': None,
        'LIB np.fft.fft': None,
    }

    for name, (function, K) in tqdm(DFT_IMPLEMENTATIONS_LIST.items()):

        N = np.asarray([2 ** k for k in range(1, K)])

        foo = []
        for i in range(len(N)):
            start = time.time()
            for j in range(R):
                function(np.random.randn(N[i], ))
            end = time.time()
            foo.append((end-start) / R)

        results[name] = (N, foo)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=100)

    for name, (test_N, r) in results.items():
        ax.plot(test_N, r, 'o-', label=name)

    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)

    ax.set_ylabel('Mean time to compute DFT [s]')
    ax.set_xlabel('Numer of samples')
    ax.legend()
    ax.grid()
    plt.show()


import numpy as np

from scipy.signal import freqz

def signal_a():
    fs = 100
    dt = 1 / fs

    t = np.arange(0, 1, dt)

    signal = 4 * np.sin(14 * np.pi * t + np.pi / 2) \
             + 3 * np.sin(8 * np.pi * t - np.pi / 2) \
             + 6 * np.sin(3 * np.pi * t + np.pi) \
             + 5 * np.sin(12 * np.pi * t - np.pi) \
             + 7 * np.sin(5 * np.pi * t - 1.5 * np.pi)
    return signal, fs

def signal_b():
    fs = 20
    dt = 1 / fs

    t1 = np.arange(0, 0.2, dt)
    t2 = np.arange(0.2, 0.4, dt)
    t3 = np.arange(0.4, 0.6, dt)
    t4 = np.arange(0.6, 0.8, dt)
    t5 = np.arange(0.8, 01.0, dt)

    signal1 = 4 * np.sin(14 * np.pi * t1 + np.pi / 2)
    signal2 = 3 * np.sin(8 * np.pi * t2 - np.pi / 2)
    signal3 = 6 * np.sin(3 * np.pi * t3 + np.pi)
    signal4 =  5 * np.sin(12 * np.pi * t4 - np.pi)
    signal5 =  7 * np.sin(5 * np.pi * t5 - 1.5 * np.pi)
    signal = np.concatenate([signal1, signal2, signal3, signal4, signal5])
    return signal, 100

def plot_DFT(X, fs, only_positive_half=True, semilog=False):


    # Number of samples in signal
    N = X.size
    X = np.where(abs(X) < 1e-10, 0, X)



    if only_positive_half == True:
        freqs = np.arange(0, N)
    else:
        freqs = np.arange(0, N) - N / 2 if N % 2 == 0 else np.arange(0, N) - (N - 1) / 2
        X_1 = np.flip(X[N:int(max(freqs)):-1])
        X_2 = X[0: int(max(freqs)) + 1]
        X = np.concatenate([X_1, X_2])
    phase = np.angle(X)
    A = abs(X) / N

    # Plot signal
    fig, (ax_amp, ax_phase) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), dpi=100)

    ax_amp.stem(freqs, A, use_line_collection=True)
    if semilog:
        ax_amp.set_yscale('log', basey=10)
    ax_amp.set_ylabel('Amplitude')
    ax_amp.set_xlabel('Frequency [Hz]')
    ax_amp.grid()

    ax_phase.stem(freqs, phase, use_line_collection=True)
    ax_phase.set_ylabel('Phase [radians]')
    ax_phase.set_xlabel('Frequency [Hz]')
    ax_phase.set_ylim([-np.pi, np.pi])
    ax_phase.grid()

    plt.tight_layout()
    plt.show()

def plot_complex(x, y):
    fig, ax = plt.subplots( figsize=(10, 5))
    ax.plot(x, y, 'o', color='black')
    ax.set_ylabel('Im')
    ax.set_xlabel('Re')
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.grid()
    plt.show()


def test_signal_a():

    signal, fs = signal_a()
    X = fft(signal)
    X_real = X.real
    X_imag = X.imag
    plot_complex(X_real, X_imag)
    plot_DFT(X, fs=fs, only_positive_half=False)

def test_signal_b():

    signal, fs = signal_b()
    X = fft(signal)
    X_real = X.real
    X_imag = X.imag
    plot_complex(X_real, X_imag)
    plot_DFT(X, fs=fs, only_positive_half=False)

def result_fft_signal_a():
    signal, fs = signal_a()


test_signal_b()