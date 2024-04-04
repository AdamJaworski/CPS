from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Design specifications for a bandpass filter
center_frequency = 96e6  # 96 MHz
bandwidth = 1e6         # 1 MHz on either side of the center frequency
sampling_rate = 200e6   # Sampling frequency (200 MHz), assumed for calculation

# Calculate the Nyquist frequency
nyquist_rate = sampling_rate / 2

# Calculate the cutoff frequencies (as a fraction of the Nyquist rate)
low_cutoff = (center_frequency - bandwidth/2) / nyquist_rate
high_cutoff = (center_frequency + bandwidth/2) / nyquist_rate

# Filter order and attenuation in the stopband (40 dB)
N, Wn = signal.buttord([low_cutoff, high_cutoff], [low_cutoff - 0.01, high_cutoff + 0.01],
                       3, 40, analog=False)

# Create a digital bandpass filter using the Butterworth design
b, a = signal.butter(N, Wn, btype='band', analog=False)

# Frequency response of the filter
w, h = signal.freqz(b, a, worN=8000)

# Plot the amplitude response of the filter
plt.figure()
plt.plot(0.5*sampling_rate*w/np.pi, 20*np.log10(abs(h)), 'b')
plt.axvline((center_frequency - bandwidth/2), color='k')
plt.axvline((center_frequency + bandwidth/2), color='k')
plt.xlim(0, 0.5*sampling_rate)
plt.title("Bandpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.show()

# Output the filter coefficients and order for verification
N, b, a
