import numpy as np
from scipy.fftpack import fft
from scipy.signal import lfilter, freqz, chirp
import matplotlib.pyplot as plt
from zad1_funcs import plot_lin_data, plot_log_data

# Parametry
fs = 48000
N = 4096
Mlen = 240
Mstep = 180
Np = 10
num_trials = 100

# Funkcja generująca sygnał typu chirp
def generate_chirp_signal(fs, N):
    t = np.arange(N) / fs
    return chirp(t, f0=0, f1=fs/2, t1=t[-1], method='linear')

# Funkcja do analizy i syntezy sygnału
def analyze_and_synthesize(x, Mlen, Mstep, Np):
    N = len(x)
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

    return x, s

# Zmienna do przechowywania sumy odpowiedzi częstotliwościowych
H_sum = None

# Powtarzanie procesu dla kilku sygnałów testowych
for _ in range(num_trials):
    x = generate_chirp_signal(fs, N)
    x, s = analyze_and_synthesize(x, Mlen, Mstep, Np)

    min_length = min(len(x), len(s))
    x = x[:min_length]
    s = s[:min_length]

    input_fft = fft(x)
    output_fft = fft(s)

    # Inicjalizacja H_sum przy pierwszej iteracji
    if H_sum is None:
        H_sum = np.zeros(min_length, dtype=complex)

    # Obliczanie odpowiedzi częstotliwościowej filtru i dodawanie do sumy
    H_sum += output_fft / input_fft

# Uśrednianie odpowiedzi częstotliwościowej
H_avg = H_sum / num_trials
frequencies = np.fft.fftfreq(min_length, 1/fs)

# Rysowanie charakterystyki amplitudowo-częstotliwościowej
plt.figure()
plt.plot(frequencies[:min_length//2], 20 * np.log10(np.abs(H_avg[:min_length//2])))
plt.title('Charakterystyka amplitudowo-częstotliwościowa filtru (średnia)')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Amplituda (dB)')
plt.grid()
plt.show()
