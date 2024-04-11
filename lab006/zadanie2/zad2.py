import scipy
from scipy.fft import fftshift
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, lfilter, butter, lfilter_zi, filtfilt

data_path = r'./data/'

keys = {
    (697, 1209): '1',
    (697, 1336): '2',
    (697, 1477): '3',
    (770, 1209): '4',
    (770, 1336): '5',
    (770, 1477): '6',
    (852, 1209): '7',
    (852, 1336): '8',
    (852, 1477): '9',
    (941, 1209): '*',
    (941, 1336): '0',
    (941, 1477): '#'
}


def get_freq(low_f, high_f) -> int:
    low_v  = np.array([697, 770, 852, 941])
    high_v = np.array([1209, 1336, 1477])

    low_min = abs(low_v - low_f)
    high_min = abs(high_v - high_f)

    low_f = low_v[np.where(low_min == np.min(low_min))][0]
    high_f = high_v[np.where(high_min == np.min(high_min))][0]

    return keys[(low_f, high_f)]


for file in range(4, 5):
    fs, data = wavfile.read(f'{data_path}s{file}.wav')
    code = ''
    for i in range(1, int(len(data) / fs)):

        fft_result = np.fft.fft(data[(i - 1) * fs: i * fs])
        fft_freq = np.fft.fftfreq(len(data[(i - 1) * fs: i * fs]), 1 / fs)

        # Taking the magnitude of the FFT result (for volume) and only the first half (due to symmetry)
        n = len(fft_result) // 2
        fft_magnitude = np.abs(fft_result[:n]) * 2 / len(data[(i - 1) * fs: i * fs])

        x = 650 # Lower frequency limit
        y = 1500  # Upper frequency limit

        highest_vol = 0
        low_f = 0
        for freq in range(680, 1000):
            if fft_magnitude[freq] > highest_vol:
                highest_vol = fft_magnitude[freq]
                low_f = freq

        highest_vol = 0
        high_f = 0
        for freq in range(1200, 1500):
            if fft_magnitude[freq] > highest_vol:
                highest_vol = fft_magnitude[freq]
                high_f = freq

        code += get_freq(low_f, high_f)

        indices = np.where((fft_freq >= x) & (fft_freq <= y))[0]
        # Plotting the Frequency vs Volume (Amplitude) graph
        plt.figure(figsize=(14, 6))
        plt.plot(fft_freq[indices], fft_magnitude[indices])
        plt.xticks(np.arange(x, y, 50))
        plt.title('Frequency vs Volume (Amplitude)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Volume (Amplitude)')
        plt.grid(True)
        plt.show()

    print(f"{file}. {code}")


# 0, 6, 5, 5, 1


