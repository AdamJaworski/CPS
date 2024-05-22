import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, freqz, welch
import matplotlib.pyplot as plt
from zad1_funcs import plot_lin_data, plot_log_data

data = ['mowa1.wav']
data_name = ["mowa1"]

for index, data_file in enumerate(data):
    # data prep
    fs, x = wavfile.read(data_file)

    N = len(x)
    Mlen = 240
    Mstep = 180
    Np = 100 #10 #8 #6 #4 #2 # Dla 100 całkiem ładnie działa
    gdzie = Mstep + 1

    lpc = []
    s = []
    ss = []
    bs = np.zeros(Np)
    Nramek = int((N - Mlen) / Mstep + 1)

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

    # Save the synthesized speech to a file
    wavfile.write(f'synthesized_{data_file}_{Np}.wav', fs, s)