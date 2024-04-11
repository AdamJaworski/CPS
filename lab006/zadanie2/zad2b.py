import numpy as np
from scipy.signal import find_peaks, lfilter, butter, lfilter_zi, filtfilt
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import fftfreq
# Load the WAV file
data_path = r'./data/'
fs, audio = wavfile.read(f'{data_path}challenge 2022.wav')

list_of_known_peaks = []
for i in range(1, int(len(audio) / 1000)):
    fft_result = np.fft.fft(audio[(i - 1) * 1000: i * 1000])
    freq = fftfreq(len(audio[(i - 1) * 1000: i * 1000]), 1 / fs)

    n = len(fft_result) // 2
    volume = np.abs(fft_result[:n]) * 2 / len(audio[(i - 1) * 1000: i * 1000])

    # plt.plot(freq, volume)
    # plt.show()

    threshold = 30
    for index, energy in enumerate(volume):
        for peak in list_of_known_peaks:
            if volume[peak[1]] >= threshold:
                continue
            else:
                list_of_known_peaks.remove(peak)
        if energy > threshold:
            if freq[index] < 500 or freq[index] > 1000:
                continue
            if list_of_known_peaks.__contains__((freq[index], index)):
                continue
            else:
                list_of_known_peaks.append((freq[index], index))
                print(f"Detected new sound with dominant frequency: {freq[index]} Hz")


# # Detect segments with energy above a threshold
# threshold = np.mean(energy) * 1.5  # Example threshold
# peaks, _ = find_peaks(energy, height=threshold)
#
# for peak in peaks:
#     start = peak * hop_length
#     end = start + frame_size
#     segment = audio[start:end]
#     fft_result = np.fft.fft(segment)
#     n = len(segment)
#     freqs = np.fft.fftfreq(n, 1/fs)[:n//2]  # Only take the first half
#     positive_fft = np.abs(fft_result)[:n//2]  # Only take the magnitude of the first half
#     dominant_freq = freqs[np.argmax(positive_fft)]
#     print(f"Detected new sound with dominant frequency: {dominant_freq} Hz")