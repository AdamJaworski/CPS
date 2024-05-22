import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


mat_contents = scipy.io.loadmat('lab08_am.mat')
signal_data = mat_contents['s7']

fs = 1000
timestep = 1/fs

signal = signal_data.flatten()
N = signal.size
t = np.linspace(0.0, N*timestep, N, endpoint=False)

fft_signal = fft(signal)
fft_freq   = fftfreq(N, d=timestep)

fft_magnitude = 2.0/N * np.abs(fft_signal[:N//2])

plt.figure(figsize=(14, 6))
plt.plot(fft_freq[:N//2], fft_magnitude)
plt.title('Spectrum of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

peaks, _ = find_peaks(fft_magnitude, height=0)
peak_freqs = fft_freq[:N//2][peaks]
peak_mags = fft_magnitude[peaks]

sorted_peak_indices = np.argsort(peak_mags)[::-1]
sorted_peak_freqs = peak_freqs[sorted_peak_indices]
sorted_peak_mags = peak_mags[sorted_peak_indices]

print(sorted_peak_freqs)
print(sorted_peak_mags)
