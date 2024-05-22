import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window


def plot_lin_data(x_data, y_data=None, label: str = None):
    if y_data is None:
        y_data = [*range(len(x_data))]

    plt.figure(figsize=(12, 6))
    plt.plot(y_data, x_data)
    if label:
        plt.title(label)
    plt.grid()
    plt.show()


def plot_log_data(x_data, y_data, label: str = None):
    plt.figure(figsize=(12, 6))
    plt.semilogy(y_data, x_data)
    if label:
        plt.title(label)
    plt.grid()
    plt.show()


def plot_freq_data(x_data, fs, window_type='hann'):
    # Normalize the signal
    x_data = x_data / np.max(np.abs(x_data))

    # Apply windowing
    window = get_window(window_type, len(x_data))
    x_data_windowed = x_data * window

    fft_results = np.fft.fft(x_data_windowed)
    frequencies = np.fft.fftfreq(len(fft_results), 1 / fs)

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_results)[:len(frequencies) // 2])
    plt.title('Sygnał w dziedzinie częstotliwości')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()