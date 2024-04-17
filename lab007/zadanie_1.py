import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Parametry
fpr = 1200  # częstotliwość próbkowania
fc = 300   # częstotliwość środkowa
df = 200   # szerokość pasma przepustowego
N = 128    # długość filtru

# Pasmo przepustowe
f1 = fc - df / 2
f2 = fc + df / 2

# Typy okien
windows = ['boxcar', 'hann', 'hamming', 'blackman', 'blackmanharris']
filters = {}

# Sygnał testowy
t = np.linspace(0, 1, fpr, endpoint=False)
x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 550 * t)
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), 1/fpr)


# Projektowanie filtrów
for window in windows:
    taps = firwin(N, [f1, f2], pass_zero=False, window=window, fs=fpr)
    filters[window] = taps
    w, h = freqz(taps, worN=8000, fs=fpr)

    # Create a new figure for each window type
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Filter characteristics - {window}')

    # Amplitude response
    plt.subplot(2, 2, 1)
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-9))
    plt.title('Amplitude Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()

    # Phase response
    plt.subplot(2, 2, 2)
    plt.plot(w, np.unwrap(np.angle(h)))
    plt.title('Phase Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid()

    # Spectrum of the original signal
    plt.subplot(2, 2, 3)
    plt.plot(freqs[:len(freqs) // 2], 20 * np.log10(np.abs(X[:len(X) // 2]) + 1e-9))
    plt.title('Spectrum of the Original Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()

    # Spectrum of the filtered signal
    plt.subplot(2, 2, 4)
    y = np.convolve(x, taps, mode='same')  # Filtering the signal
    Y = np.fft.fft(y)
    plt.plot(freqs[:len(Y) // 2], 20 * np.log10(np.abs(Y[:len(Y) // 2]) + 1e-9), label=f'{window}')
    plt.title('Spectrum of the Filtered Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.legend()
    plt.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.show()



