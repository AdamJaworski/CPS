import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, freqz, welch
import matplotlib.pyplot as plt
from zad1_funcs import plot_lin_data, plot_log_data, plot_freq_data

data = ['A.wav', 'S.wav', 'mowa1.wav']
data_name = ["Dźwięczna", "Bezdźwięczna", "mowa1"]

for index, data_file in enumerate(data):
    # data prep
    fs, x = wavfile.read(data_file)

    if len(x.shape) > 1:
        x_mono = x.mean(axis=1)
    else:
        x_mono = x

    # display data signal
    plot_lin_data(x, label=f'signal {data_name[index]}')
    plot_freq_data(x_mono, fs)

    # display PSD of signal
    frequencies, psd = welch(x_mono, fs, nperseg=1024)
    plot_log_data(psd, frequencies, 'Power Spectral Density')

    N = len(x)
    Mlen = 240
    Mstep = 180
    Np = 10
    gdzie = Mstep + 1

    lpc = []
    s = []
    ss = []
    bs = np.zeros(Np)
    Nramek = int((N - Mlen) / Mstep + 1)

    # Pre-emp
    x = lfilter([1, -0.9735], 1, x)

    if len(x.shape) > 1:
        x_mono = x.mean(axis=1)
    else:
        x_mono = x

    # display data signal
    plot_lin_data(x, label=f'signal {data_name[index]} Pre-emp')

    # display PSD of signal
    frequencies, psd = welch(x_mono, fs, nperseg=1024)
    plot_log_data(psd, frequencies, 'Power Spectral Density Pre-emp')

    for nr in range(Nramek):
        # Get the next fragment of the signal
        n = np.arange((nr * Mstep), (nr * Mstep) + Mlen)
        bx = x[n]

        # ANALYSIS - determine model parameters
        bx = bx - np.mean(bx)  # Remove mean value
        r = np.array([np.sum(bx[:Mlen - k] * bx[k:]) for k in range(Mlen)])  # Autocorrelation function

        # Find the maximum of the autocorrelation function
        offset = 20
        rmax = np.max(r[offset:])
        imax = np.argmax(r == rmax)
        T = imax if rmax > 0.35 * r[0] else 0  # Voiced/unvoiced decision
        print(f"T: {T}, imax: {imax}")

        rr = r[1:Np + 1]
        R = np.array([r[m::-1].tolist() + r[1:Np - m].tolist() for m in range(Np)])  # Autocorrelation matrix
        a = -np.linalg.inv(R).dot(rr)  # Compute prediction filter coefficients
        wzm = r[0] + rr.dot(a)  # Compute gain
        w, h = freqz(1, np.concatenate(([1], a)))  # Compute frequency response

        # SYNTHESIS - reconstruct based on parameters
        if T != 0:
            gdzie = gdzie - Mstep  # Move voiced excitation
        for n in range(Mstep):
            if T == 0:      # sprawdzenie czy moza jest dźwięczna
                pob = 2 * (np.random.rand() - 0.5)
                gdzie = int(1.5 * Mstep + 1)  # Noise excitation
            else:
                if n == gdzie:
                    pob = 1
                    gdzie = gdzie + T  # Voiced excitation
                else:
                    pob = 0
            ss.append(wzm * pob - bs.dot(a))  # Filter "synthetic" excitation
            bs = np.concatenate(([ss[-1]], bs[:-1]))  # Shift output buffer

        s.extend(ss)  # Store synthesized speech fragment
        ss = []  # Reset fragment buffer

    # Normalize the synthesized signal to the range of int16
    s = np.array(s)
    s = s / np.max(np.abs(s))  # Normalize to the range -1 to 1
    s = (s * 32767).astype(np.int16)  # Scale to int16 range

    plt.plot(s)
    plt.title('Synthesized Speech')
    plt.show()

    if len(s.shape) > 1:
        s_mono = s.mean(axis=1)
    else:
        s_mono = s
    plot_freq_data(s_mono, fs)
    # Save the synthesized speech to a file
    wavfile.write(f'synthesized_{data_file}.wav', fs, s)
